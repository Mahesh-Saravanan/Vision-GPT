"""
Flickr30k DataLoader for VisionGPT2 image captioning.

LOADING STRATEGY (tried in order)
-----------------------------------
1. Hub-parquet   — loads nlphuji/flickr30k parquet files directly via
                   huggingface_hub, completely bypassing the broken
                   flickr30k.py script.  Works with datasets >= 3.0.
                   No local files needed; images download on first run.

2. Local-Karpathy — reads a local dataset_flickr30k.json (Karpathy split)
                    plus a local flickr30k-images/ directory.
                    Fastest after first run; no HuggingFace dependency.

The constructor tries strategy 1 automatically when local files are absent.
Pass explicit karpathy_json + image_root to force strategy 2.

GPT-2 tokenisation
-------------------
  pad_token = eos_token  (<|endoftext|>, id 50256)
  EOS appended to every caption; padding positions → label -100
  so CrossEntropyLoss ignores them without masking real EOS tokens.
"""

import os
import io
import json
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Strategy 1 – load directly from Hub parquet files
# ---------------------------------------------------------------------------

def _load_from_hub(split: str) -> list:
    """
    Downloads nlphuji/flickr30k parquet files one at a time via huggingface_hub
    and returns a flat list of {'image': PIL.Image, 'caption': str} dicts.

    This completely sidesteps the datasets library script-detection error
    because we never call load_dataset() at all.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Install with:  pip install huggingface_hub pyarrow"
        ) from e

    print(f"  Scanning nlphuji/flickr30k on the Hub for parquet files …")
    all_files    = list(list_repo_files("nlphuji/flickr30k", repo_type="dataset"))
    parquet_files = sorted(f for f in all_files if f.endswith(".parquet"))

    if not parquet_files:
        raise RuntimeError("No parquet files found in nlphuji/flickr30k.")

    samples = []
    for pf in parquet_files:
        print(f"  Downloading {pf} …")
        local_path = hf_hub_download(
            repo_id="nlphuji/flickr30k",
            filename=pf,
            repo_type="dataset",
        )
        table = pq.read_table(local_path)
        df    = table.to_pydict()

        # Determine column names defensively
        img_col     = next((c for c in df if c in ("image", "img", "pixel_values")), None)
        cap_col     = next((c for c in df if c in ("caption", "captions", "sentences")), None)
        split_col   = next((c for c in df if c in ("split", "set", "subset")), None)

        if img_col is None or cap_col is None:
            raise RuntimeError(
                f"Unexpected parquet schema. Columns found: {list(df.keys())}"
            )

        n = len(df[img_col])
        for i in range(n):
            # Filter by split when a split column exists
            if split_col is not None and df[split_col][i] != split:
                continue

            # Decode image: HuggingFace stores images as {'bytes': b'...', 'path': '...'}
            raw_img = df[img_col][i]
            if isinstance(raw_img, dict):
                img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
            elif isinstance(raw_img, bytes):
                img = Image.open(io.BytesIO(raw_img)).convert("RGB")
            else:
                img = raw_img.convert("RGB")  # already a PIL Image

            captions = df[cap_col][i]
            if isinstance(captions, str):
                captions = [captions]

            for caption in captions:
                samples.append({"image": img, "caption": caption})

    if not samples:
        raise RuntimeError(
            f"Hub parquet load succeeded but found 0 samples for split='{split}'.\n"
            "The dataset may not include a 'split' column — every row was filtered out.\n"
            "Try setting split_col=None logic or use the local Karpathy JSON instead."
        )

    return samples


# ---------------------------------------------------------------------------
# Strategy 2 – load from local Karpathy split JSON + images directory
# ---------------------------------------------------------------------------

KARPATHY_URL = (
    "https://cs.stanford.edu/people/karpathy/"
    "deepimagesent/caption_datasets.zip"
)


def download_karpathy_json(dest_dir: str = ".") -> str:
    """
    Downloads the Karpathy split caption ZIP from Stanford and extracts
    dataset_flickr30k.json into dest_dir.  Returns the path to the JSON.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path  = os.path.join(dest_dir, "caption_datasets.zip")
    json_path = os.path.join(dest_dir, "dataset_flickr30k.json")

    if os.path.isfile(json_path):
        return json_path

    print(f"  Downloading Karpathy annotations from Stanford …")
    urllib.request.urlretrieve(KARPATHY_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("dataset_flickr30k.json", dest_dir)

    os.remove(zip_path)
    print(f"  Saved → {json_path}")
    return json_path


def _load_from_local(split: str, karpathy_json: str, image_root: str) -> list:
    """
    Reads dataset_flickr30k.json and loads images from image_root/flickr30k-images/.
    Returns a flat list of {'image': PIL.Image, 'caption': str} dicts.
    """
    with open(karpathy_json, "r") as f:
        data = json.load(f)

    samples = []
    for entry in data["images"]:
        if entry["split"] != split:
            continue
        img_path = os.path.join(
            image_root,
            entry.get("filepath", "flickr30k-images"),
            entry["filename"],
        )
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"\nImage not found: {img_path}\n\n"
                "Please download the Flickr30k images from the official source:\n"
                "  https://shannon.cs.illinois.edu/DenotationGraph/\n"
                "and place them so the path is:\n"
                "  <image_root>/flickr30k-images/<filename>.jpg\n\n"
                "Alternatively, remove karpathy_json / image_root arguments and let\n"
                "the dataloader fetch everything from the HuggingFace Hub automatically."
            )
        for sentence in entry["sentences"]:
            samples.append({"image": img, "caption": sentence["raw"]})

    return samples


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset for image captioning.

    Tries Hub-parquet loading automatically unless local paths are given.

    Args:
        split          : 'train', 'val', or 'test'
        karpathy_json  : path to dataset_flickr30k.json  (local strategy)
        image_root     : directory containing flickr30k-images/  (local strategy)
        max_length     : max tokenised caption length
        image_size     : ViT input resolution (default 224)
        tokenizer      : GPT2Tokenizer; created from 'gpt2' if None
    """

    def __init__(
        self,
        split:         str  = "train",
        karpathy_json: str  = None,
        image_root:    str  = None,
        max_length:    int  = 64,
        image_size:    int  = 224,
        tokenizer            = None,
    ):
        self.max_length = max_length
        self.transform  = get_image_transform(image_size)

        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # Choose loading strategy
        local_json_given = karpathy_json is not None and os.path.isfile(karpathy_json)
        local_img_given  = image_root is not None

        if local_json_given and local_img_given:
            print(f"Loading Flickr30k ({split}) from local Karpathy JSON …")
            self.samples = _load_from_local(split, karpathy_json, image_root)
        else:
            print(f"Loading Flickr30k ({split}) from HuggingFace Hub (parquet) …")
            try:
                self.samples = _load_from_hub(split)
            except Exception as e:
                raise RuntimeError(
                    f"\nHub parquet loading failed: {e}\n\n"
                    "Options to fix:\n"
                    "  A) Downgrade datasets:  pip install 'datasets>=2.14,<3.0'\n"
                    "  B) Provide local files:\n"
                    "       1. Captions (auto-download):\n"
                    "            python -c \"from dataloader import download_karpathy_json; download_karpathy_json()\"\n"
                    "       2. Images: https://shannon.cs.illinois.edu/DenotationGraph/\n"
                    "       3. Pass karpathy_json='dataset_flickr30k.json' + image_root='.' to get_dataloader()\n"
                ) from e

        print(f"  → {len(self.samples):,} (image, caption) pairs in '{split}' split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample  = self.samples[idx]
        caption = sample["caption"]

        pixel_values = self.transform(sample["image"])  # [3, 224, 224]

        # Append EOS so the model learns to terminate generation
        text = caption + self.tokenizer.eos_token

        tokens = self.tokenizer(
            text,
            max_length=self.max_length + 1,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        full_ids  = tokens["input_ids"].squeeze(0)
        full_mask = tokens["attention_mask"].squeeze(0)

        input_ids  = full_ids[:-1].clone()
        target_ids = full_ids[1:].clone()
        loss_mask  = full_mask[1:]

        target_ids[loss_mask == 0] = -100   # ignore padding in loss

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "target_ids":   target_ids,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    split:         str  = "train",
    karpathy_json: str  = None,
    image_root:    str  = None,
    batch_size:    int  = 32,
    num_workers:   int  = 4,
    max_length:    int  = 64,
    image_size:    int  = 224,
    tokenizer            = None,
) -> DataLoader:
    dataset = Flickr30kDataset(
        split=split,
        karpathy_json=karpathy_json,
        image_root=image_root,
        max_length=max_length,
        image_size=image_size,
        tokenizer=tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = get_dataloader(split="val", batch_size=4, num_workers=0)
    batch  = next(iter(loader))
    print("pixel_values:", batch["pixel_values"].shape)
    print("input_ids:   ", batch["input_ids"].shape)
    print("target_ids:  ", batch["target_ids"].shape)
    print("Sanity check passed.")
