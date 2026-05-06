"""
Flickr30k DataLoader for VisionGPT2.

Supported data sources (tried in this order):
  1. Local folder  — fastest; used automatically when Data/ exists next to train.py
  2. HuggingFace Hub — downloads on first run, cached to ~/.cache/flickr30k/

LOCAL FORMAT (Data/)
--------------------
  Data/
  ├── Images/
  │   ├── 000092795.jpg
  │   ├── 10002456.jpg
  │   └── ...
  └── captions.txt          one line per caption:
                             000092795.jpg, Two friends enjoy time spent together .
                             10002456.jpg, Several men in hard hats are operating ...
                             10002456.jpg, Workers look down from up above ...

Since captions.txt has no split column, a deterministic split is created
from the sorted list of unique image filenames:
  train  — first  (N - val_size - test_size) unique images
  val    — next   val_size  images  (default 1 000)
  test   — last   test_size images  (default 1 000)

GPT-2 TOKENISATION
------------------
  pad_token = eos_token  (<|endoftext|>, id 50 256)
  EOS appended to every caption; padding positions → label -100
"""

import os
import csv
import io
import ast
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Default local data directory (relative to wherever train.py is run from)
_DEFAULT_DATA_DIR   = "Data"
_DEFAULT_IMAGES_DIR = os.path.join(_DEFAULT_DATA_DIR, "Images")
_DEFAULT_CAPTIONS   = os.path.join(_DEFAULT_DATA_DIR, "captions.txt")

# Hub extraction cache
_DEFAULT_EXTRACT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "flickr30k")


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------------------------------------------------
# Strategy 1  —  local Data/ folder
# -----------------------------------------------------------------------

def _load_from_local_folder(
    split:      str,
    data_dir:   str = _DEFAULT_DATA_DIR,
    val_size:   int = 1000,
    test_size:  int = 1000,
) -> list:
    """
    Parses Data/captions.txt and Data/Images/.

    captions.txt format  (one caption per line, comma after filename):
        000092795.jpg, Two friends enjoy time spent together .
        10002456.jpg,  Several men in hard hats are operating a giant pulley system .

    Returns a flat list of {'image_path': str, 'caption': str}.
    """
    captions_path = os.path.join(data_dir, "captions.txt")
    images_dir    = os.path.join(data_dir, "Images")

    if not os.path.isfile(captions_path):
        raise FileNotFoundError(f"captions.txt not found at: {captions_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    # --- Parse captions ---
    # Use split(", ", 1) so commas inside caption text are preserved.
    image_captions: dict = {}
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split only on the first ", " to preserve commas in captions
            if ", " in line:
                filename, caption = line.split(", ", 1)
            elif "," in line:
                filename, caption = line.split(",", 1)
            else:
                continue
            filename = filename.strip()
            caption  = caption.strip()
            image_captions.setdefault(filename, []).append(caption)

    # --- Deterministic train/val/test split on sorted filenames ---
    all_images = sorted(image_captions.keys())
    n          = len(all_images)

    if n < val_size + test_size:
        raise ValueError(
            f"Only {n} unique images found, but val_size+test_size="
            f"{val_size + test_size}.  Reduce val_size / test_size."
        )

    # Sorted order: first (n - val - test) → train, then val, then test
    train_set = set(all_images[:n - val_size - test_size])
    val_set   = set(all_images[n - val_size - test_size : n - test_size])
    test_set  = set(all_images[n - test_size:])
    target    = {"train": train_set, "val": val_set, "test": test_set}[split]

    samples = []
    for filename in all_images:
        if filename not in target:
            continue
        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"Image listed in captions.txt not found on disk: {img_path}"
            )
        for caption in image_captions[filename]:
            samples.append({"image_path": img_path, "caption": caption})

    return samples


# -----------------------------------------------------------------------
# Strategy 2  —  HuggingFace Hub (nlphuji/flickr30k)
# -----------------------------------------------------------------------

def _ensure_images_extracted(extract_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    images_dir = os.path.join(extract_dir, "flickr30k-images")
    if os.path.isdir(images_dir) and len(os.listdir(images_dir)) >= 31000:
        return images_dir

    print("  Downloading flickr30k-images.zip (~9 GB) — cached after first run …")
    zip_cached = hf_hub_download(
        repo_id="nlphuji/flickr30k",
        filename="flickr30k-images.zip",
        repo_type="dataset",
    )
    os.makedirs(extract_dir, exist_ok=True)
    print(f"  Extracting to {extract_dir} …")
    with zipfile.ZipFile(zip_cached, "r") as zf:
        zf.extractall(extract_dir)
    print(f"  Done → {images_dir}")
    return images_dir


def _load_from_hub(split: str, extract_dir: str = _DEFAULT_EXTRACT_DIR) -> list:
    from huggingface_hub import hf_hub_download

    print("  Downloading flickr_annotations_30k.csv …")
    csv_path   = hf_hub_download(
        repo_id="nlphuji/flickr30k",
        filename="flickr_annotations_30k.csv",
        repo_type="dataset",
    )
    images_dir = _ensure_images_extracted(extract_dir)

    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            img_path = os.path.join(images_dir, row["filename"])
            for cap in ast.literal_eval(row["raw"]):
                samples.append({"image_path": img_path, "caption": str(cap).strip()})

    return samples


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class Flickr30kDataset(Dataset):
    """
    Args:
        split       : 'train', 'val', or 'test'
        data_dir    : path to folder containing Images/ and captions.txt
                      (auto-detected as 'Data/' if it exists)
        val_size    : images reserved for validation  (local mode only)
        test_size   : images reserved for testing     (local mode only)
        extract_dir : where to extract Hub ZIP        (Hub mode only)
        max_length  : max tokenised caption length
        image_size  : ViT input resolution
        tokenizer   : GPT2Tokenizer; built from 'gpt2' if None
    """

    def __init__(
        self,
        split:       str  = "train",
        data_dir:    str  = None,
        val_size:    int  = 1000,
        test_size:   int  = 1000,
        extract_dir: str  = _DEFAULT_EXTRACT_DIR,
        max_length:  int  = 64,
        image_size:  int  = 224,
        tokenizer         = None,
    ):
        self.max_length = max_length
        self.transform  = get_image_transform(image_size)

        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # Auto-detect local Data/ folder
        if data_dir is None and os.path.isdir(_DEFAULT_DATA_DIR):
            data_dir = _DEFAULT_DATA_DIR

        if data_dir is not None:
            print(f"Loading Flickr30k ({split}) from local folder: {data_dir} …")
            self.samples = _load_from_local_folder(
                split, data_dir, val_size, test_size
            )
        else:
            print(f"Loading Flickr30k ({split}) from HuggingFace Hub …")
            self.samples = _load_from_hub(split, extract_dir)

        if not self.samples:
            raise RuntimeError(f"0 samples found for split='{split}'.")

        print(f"  → {len(self.samples):,} (image, caption) pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample  = self.samples[idx]
        caption = sample["caption"]

        img          = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(img)          # [3, 224, 224]

        text   = caption + self.tokenizer.eos_token
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
        target_ids[full_mask[1:] == 0] = -100      # ignore padding in loss

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "target_ids":   target_ids,
        }


# -----------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------

def get_dataloader(
    split:       str  = "train",
    data_dir:    str  = None,
    val_size:    int  = 1000,
    test_size:   int  = 1000,
    extract_dir: str  = _DEFAULT_EXTRACT_DIR,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    max_length:  int  = 64,
    image_size:  int  = 224,
    tokenizer         = None,
) -> DataLoader:
    ds = Flickr30kDataset(
        split=split,
        data_dir=data_dir,
        val_size=val_size,
        test_size=test_size,
        extract_dir=extract_dir,
        max_length=max_length,
        image_size=image_size,
        tokenizer=tokenizer,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# -----------------------------------------------------------------------
# Sanity check:  python dataloader.py
# -----------------------------------------------------------------------

if __name__ == "__main__":
    loader = get_dataloader(split="val", batch_size=4, num_workers=0)
    batch  = next(iter(loader))
    print("pixel_values :", batch["pixel_values"].shape)   # [4, 3, 224, 224]
    print("input_ids    :", batch["input_ids"].shape)      # [4, 64]
    print("target_ids   :", batch["target_ids"].shape)     # [4, 64]
    print("OK")
