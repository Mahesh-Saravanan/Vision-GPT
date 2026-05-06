"""
Flickr30k DataLoader for VisionGPT2.

No dependency on the `datasets` library.
Uses huggingface_hub + stdlib (csv, zipfile, ast) only.

CSV schema of flickr_annotations_30k.csv (confirmed from repo):
    raw       – JSON-encoded list of 5 caption strings
    sentids   – JSON-encoded list of sentence IDs
    split     – "train" | "val" | "test"  (split already in the file)
    filename  – image filename, e.g. "1000092795.jpg"
    img_id    – integer image ID

First-run behaviour
-------------------
  1. flickr_annotations_30k.csv  (~5 MB)  → downloaded to HF cache
  2. flickr30k-images.zip        (~9 GB)  → downloaded to HF cache, then
                                             extracted once to ~/.cache/flickr30k/
  Both are cached permanently; subsequent runs skip all downloads.

Memory design
-------------
  self.samples stores (image_path, caption) strings only.
  PIL images are opened on demand inside __getitem__ — no images in RAM
  at dataset construction time.
"""

import os
import io
import ast
import csv
import json
import zipfile
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Where extracted images are cached locally
_DEFAULT_EXTRACT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "flickr30k")


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Hub loader — downloads CSV + ZIP, extracts once, returns list of
# {'image_path': str, 'caption': str} dicts
# ---------------------------------------------------------------------------

def _ensure_images_extracted(extract_dir: str) -> str:
    """
    Returns the path to the extracted flickr30k-images/ directory.
    Downloads and extracts flickr30k-images.zip on first call.
    """
    from huggingface_hub import hf_hub_download

    images_dir = os.path.join(extract_dir, "flickr30k-images")

    # Check if already extracted (needs at least 31 000 image files)
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

    print(f"  Extraction complete → {images_dir}")
    return images_dir


def _load_from_hub(split: str, extract_dir: str = _DEFAULT_EXTRACT_DIR) -> list:
    """
    Downloads flickr_annotations_30k.csv and flickr30k-images.zip from
    nlphuji/flickr30k via hf_hub_download (no `datasets` library needed),
    then returns a flat list of {'image_path': str, 'caption': str} dicts.
    """
    from huggingface_hub import hf_hub_download

    # --- Step 1: annotations CSV (small, fast) ---
    print("  Downloading flickr_annotations_30k.csv …")
    csv_path = hf_hub_download(
        repo_id="nlphuji/flickr30k",
        filename="flickr_annotations_30k.csv",
        repo_type="dataset",
    )

    # --- Step 2: images ZIP (large, extracted once) ---
    images_dir = _ensure_images_extracted(extract_dir)

    # --- Step 3: parse CSV and build sample list ---
    # raw column: JSON list of 5 caption strings, e.g.
    #   '["Two dogs play.", "A dog runs.", ...]'
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            img_path = os.path.join(images_dir, row["filename"])
            captions = ast.literal_eval(row["raw"])   # list of 5 strings
            for cap in captions:
                samples.append({
                    "image_path": img_path,
                    "caption":    str(cap).strip(),
                })

    return samples


# ---------------------------------------------------------------------------
# Local Karpathy loader (optional, faster if you already have the files)
# ---------------------------------------------------------------------------

def _load_from_local(split: str, karpathy_json: str, image_root: str) -> list:
    """
    Reads dataset_flickr30k.json (Karpathy split) + local image directory.
    Returns list of {'image_path': str, 'caption': str}.
    """
    with open(karpathy_json, encoding="utf-8") as f:
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
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                "Make sure image_root contains a 'flickr30k-images/' subdirectory."
            )
        for sentence in entry["sentences"]:
            samples.append({
                "image_path": img_path,
                "caption":    sentence["raw"].strip(),
            })

    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset. Images are loaded from disk on demand — no images
    held in RAM during construction.

    Args:
        split          : 'train', 'val', or 'test'
        karpathy_json  : path to dataset_flickr30k.json  (activates local mode)
        image_root     : directory containing flickr30k-images/  (local mode)
        extract_dir    : where to extract the Hub ZIP (default ~/.cache/flickr30k)
        max_length     : max tokenised caption length
        image_size     : ViT input resolution (default 224)
        tokenizer      : GPT2Tokenizer; built from 'gpt2' if None
    """

    def __init__(
        self,
        split:         str = "train",
        karpathy_json: str = None,
        image_root:    str = None,
        extract_dir:   str = _DEFAULT_EXTRACT_DIR,
        max_length:    int = 64,
        image_size:    int = 224,
        tokenizer          = None,
    ):
        self.max_length = max_length
        self.transform  = get_image_transform(image_size)

        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        use_local = (
            karpathy_json is not None
            and os.path.isfile(karpathy_json)
            and image_root is not None
        )

        if use_local:
            print(f"Loading Flickr30k ({split}) from local Karpathy JSON …")
            self.samples = _load_from_local(split, karpathy_json, image_root)
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

        # Load image from disk on demand
        img          = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(img)   # [3, 224, 224]

        # Tokenise: append EOS so model learns to terminate
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
        target_ids[full_mask[1:] == 0] = -100   # ignore padding in loss

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "target_ids":   target_ids,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    split:         str = "train",
    karpathy_json: str = None,
    image_root:    str = None,
    extract_dir:   str = _DEFAULT_EXTRACT_DIR,
    batch_size:    int = 32,
    num_workers:   int = 4,
    max_length:    int = 64,
    image_size:    int = 224,
    tokenizer          = None,
) -> DataLoader:
    ds = Flickr30kDataset(
        split=split,
        karpathy_json=karpathy_json,
        image_root=image_root,
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


# ---------------------------------------------------------------------------
# Sanity check:  python dataloader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = get_dataloader(split="val", batch_size=4, num_workers=0)
    batch  = next(iter(loader))
    print("pixel_values:", batch["pixel_values"].shape)   # [4, 3, 224, 224]
    print("input_ids:   ", batch["input_ids"].shape)      # [4, 64]
    print("target_ids:  ", batch["target_ids"].shape)     # [4, 64]
    print("OK")
