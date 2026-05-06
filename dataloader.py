"""
Flickr30k DataLoader for VisionGPT2 image captioning.

WHY THE REWRITE
---------------
`datasets >= 3.0` removed all custom Python-script support. The HuggingFace repo
`nlphuji/flickr30k` still ships a `flickr30k.py` loader, so `load_dataset()` raises:
    RuntimeError: Dataset scripts are no longer supported, but found flickr30k.py

Quick alternative (one-liner):
    pip install "datasets>=2.14,<3.0"

Code-level fix (this file):
    We load directly from the standard Karpathy split JSON + a local images folder,
    which is the format used in virtually all Flickr30k research.

HOW TO GET THE LOCAL FILES
--------------------------
1. Download Flickr30k images (requires a free Flickr30k license):
       https://shannon.cs.illinois.edu/DenotationGraph/

2. Download the Karpathy split annotations (public, no login needed):
       wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
       unzip caption_datasets.zip          # gives dataset_flickr30k.json

3. Point the dataloader at both paths:
       get_dataloader(
           split="train",
           karpathy_json="/data/flickr30k/dataset_flickr30k.json",
           image_root="/data/flickr30k",   # contains flickr30k-images/ subfolder
       )

GPT-2 tokenisation trick
------------------------
- GPT-2Tokenizer has no native pad token. We set pad_token = eos_token.
- EOS is appended to every caption so the model learns sentence endings.
- Padding positions get label = -100 so CrossEntropyLoss ignores them.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer
from PIL import Image


# ImageNet stats used by the ViT feature extractor
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Default paths — override via function arguments
DEFAULT_KARPATHY_JSON = "dataset_flickr30k.json"
DEFAULT_IMAGE_ROOT    = "."   # expects <image_root>/flickr30k-images/<filename>


# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------

def get_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset loaded from local Karpathy-split JSON + images directory.

    The Karpathy split JSON has the structure:
        {
          "images": [
            {
              "split":     "train",          # "train" | "val" | "test"
              "filename":  "1000092795.jpg",
              "filepath":  "flickr30k-images",
              "sentences": [
                {"raw": "Two dogs play.", ...},
                ...                          # 5 sentences per image
              ]
            }, ...
          ]
        }

    Each image has 5 captions. We flatten into individual (image, caption) pairs
    so every pair is a unique training sample (up to 145 000 for train).

    Args:
        split          : 'train', 'val', or 'test'
        karpathy_json  : path to dataset_flickr30k.json
        image_root     : directory that contains the flickr30k-images/ subfolder
        max_length     : maximum tokenised caption length (including appended EOS)
        image_size     : ViT input resolution (default 224)
        tokenizer      : GPT2Tokenizer instance; created from 'gpt2' if None
    """

    def __init__(
        self,
        split:         str  = "train",
        karpathy_json: str  = DEFAULT_KARPATHY_JSON,
        image_root:    str  = DEFAULT_IMAGE_ROOT,
        max_length:    int  = 64,
        image_size:    int  = 224,
        tokenizer            = None,
    ):
        self.max_length = max_length
        self.image_root = image_root
        self.transform  = get_image_transform(image_size)

        # GPT-2 tokeniser — pad with eos so batches can be collated
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # Load Karpathy JSON
        if not os.path.isfile(karpathy_json):
            raise FileNotFoundError(
                f"\nKarpathy JSON not found: '{karpathy_json}'\n\n"
                "Download it with:\n"
                "  wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip\n"
                "  unzip caption_datasets.zip\n"
                "Then pass karpathy_json='<path>/dataset_flickr30k.json' to get_dataloader().\n\n"
                "Alternatively, downgrade the datasets library:\n"
                "  pip install 'datasets>=2.14,<3.0'\n"
                "and revert to the HuggingFace loader."
            )

        print(f"Loading Flickr30k ({split}) from {karpathy_json} …")
        with open(karpathy_json, "r") as f:
            data = json.load(f)

        # Flatten into (image_path, caption) pairs
        self.samples = []
        for entry in data["images"]:
            if entry["split"] != split:
                continue
            img_path = os.path.join(image_root, entry.get("filepath", "flickr30k-images"), entry["filename"])
            for sentence in entry["sentences"]:
                self.samples.append({
                    "image_path": img_path,
                    "caption":    sentence["raw"],
                })

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split='{split}' in {karpathy_json}")

        print(f"  → {len(self.samples):,} (image, caption) pairs in '{split}' split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample  = self.samples[idx]
        caption = sample["caption"]

        # Load and preprocess image
        img          = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(img)  # [3, 224, 224]

        # Append EOS so the model learns to terminate generation
        text = caption + self.tokenizer.eos_token

        tokens = self.tokenizer(
            text,
            max_length=self.max_length + 1,  # +1 because we shift to get input / target
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        full_ids  = tokens["input_ids"].squeeze(0)       # [max_length + 1]
        full_mask = tokens["attention_mask"].squeeze(0)  # [max_length + 1]

        # Teacher-forcing shift:
        #   input_ids  = [w1, w2, …, wN]     (all but the last position)
        #   target_ids = [w2, …, wN, EOS]    (all but the first position)
        input_ids  = full_ids[:-1].clone()   # [max_length]
        target_ids = full_ids[1:].clone()    # [max_length]
        loss_mask  = full_mask[1:]           # 1 for real tokens, 0 for padding

        # Padding positions → -100 so CrossEntropyLoss skips them
        target_ids[loss_mask == 0] = -100

        return {
            "pixel_values": pixel_values,  # [3, 224, 224]
            "input_ids":    input_ids,      # [max_length]
            "target_ids":   target_ids,     # [max_length]  (−100 at padding positions)
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    split:         str  = "train",
    karpathy_json: str  = DEFAULT_KARPATHY_JSON,
    image_root:    str  = DEFAULT_IMAGE_ROOT,
    batch_size:    int  = 32,
    num_workers:   int  = 4,
    max_length:    int  = 64,
    image_size:    int  = 224,
    tokenizer            = None,
) -> DataLoader:
    """
    Returns a DataLoader for the requested Flickr30k split.

    Example:
        loader = get_dataloader(
            split="train",
            karpathy_json="/data/flickr30k/dataset_flickr30k.json",
            image_root="/data/flickr30k",
            batch_size=32,
        )
    """
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
    import sys
    json_path  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KARPATHY_JSON
    img_root   = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE_ROOT

    loader = get_dataloader(
        split="val", batch_size=4, num_workers=0,
        karpathy_json=json_path, image_root=img_root,
    )
    batch = next(iter(loader))
    print("pixel_values:", batch["pixel_values"].shape)   # [4, 3, 224, 224]
    print("input_ids:   ", batch["input_ids"].shape)      # [4, 64]
    print("target_ids:  ", batch["target_ids"].shape)     # [4, 64]
    print("Sanity check passed.")
