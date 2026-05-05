"""
Flickr30k DataLoader for VisionGPT2 image captioning.

Dataset notes:
- Flickr30k has 31 000 images, each with 5 human-written captions.
- We flatten the dataset so every (image, caption) pair is a unique training sample
  → 31 000 images × 5 captions = up to 155 000 samples for train.
- The HuggingFace hub hosts all splits in a single "test" split with a 'split' column
  that labels each row as 'train', 'val', or 'test'.

GPT-2 tokenisation trick:
- GPT-2Tokenizer has no native pad token.  We set pad_token = eos_token (<|endoftext|>, id=50256).
- We also append the eos_token to each caption so the model learns where sentences end.
- Positions that are pure padding carry label = -100 so they are ignored by CrossEntropyLoss.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer
from datasets import load_dataset
from PIL import Image


# ImageNet mean/std used by the ViT feature extractor
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class Flickr30kDataset(Dataset):
    """
    Loads Flickr30k from the HuggingFace hub ('nlphuji/flickr30k').
    Each image has 5 captions; we flatten into individual (image, caption) pairs.

    Args:
        split       : 'train', 'val', or 'test'
        max_length  : maximum tokenised caption length (including appended EOS)
        image_size  : input resolution for ViT (default 224)
        tokenizer   : a GPT2Tokenizer instance (created inside if None)
    """

    def __init__(
        self,
        split:      str  = 'train',
        max_length: int  = 64,
        image_size: int  = 224,
        tokenizer        = None,
    ):
        self.max_length = max_length
        self.transform  = get_image_transform(image_size)

        # Setup tokenizer: pad with eos so collate_fn can batch sequences
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token  # critical for GPT-2 batching
        self.tokenizer = tokenizer

        # Load dataset (all splits live in the HuggingFace 'test' split)
        print(f"Loading Flickr30k ({split}) from HuggingFace hub …")
        raw = load_dataset("nlphuji/flickr30k", split="test")
        raw = raw.filter(lambda x: x["split"] == split)

        # Flatten: one row per (image, caption) pair
        self.samples = []
        for item in raw:
            img = item["image"]  # PIL Image
            for caption in item["caption"]:  # list of 5 strings
                self.samples.append({"image": img, "caption": caption})

        print(f"  → {len(self.samples):,} (image, caption) pairs in {split} split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample  = self.samples[idx]
        caption = sample["caption"]

        # Image preprocessing
        pixel_values = self.transform(sample["image"].convert("RGB"))  # [3, 224, 224]

        # Append EOS so the model learns to generate an end token
        text = caption + self.tokenizer.eos_token

        tokens = self.tokenizer(
            text,
            max_length=self.max_length + 1,  # +1 because we shift for input / target
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        full_ids  = tokens["input_ids"].squeeze(0)       # [max_length+1]
        full_mask = tokens["attention_mask"].squeeze(0)  # [max_length+1]

        # Teacher-forcing split:
        #   input_ids  : [CLS / first_word, …, last_word]   (all but last position)
        #   target_ids : [second_word, …, EOS / last_word]  (all but first position)
        input_ids  = full_ids[:-1].clone()     # [max_length]
        target_ids = full_ids[1:].clone()      # [max_length]
        loss_mask  = full_mask[1:]             # 1 = real token, 0 = padding

        # Mark padding positions as -100 → CrossEntropyLoss ignores them
        target_ids[loss_mask == 0] = -100

        return {
            "pixel_values": pixel_values,   # [3, 224, 224]
            "input_ids":    input_ids,       # [max_length]
            "target_ids":   target_ids,      # [max_length]  (-100 at padding positions)
        }


def get_dataloader(
    split:       str  = "train",
    batch_size:  int  = 32,
    num_workers: int  = 4,
    max_length:  int  = 64,
    image_size:  int  = 224,
    tokenizer        = None,
) -> DataLoader:
    dataset = Flickr30kDataset(
        split=split,
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
        drop_last=(split == "train"),  # avoid incomplete batches during training
    )


# Quick sanity check
if __name__ == "__main__":
    loader = get_dataloader(split="val", batch_size=4, num_workers=0)
    batch  = next(iter(loader))
    print("pixel_values:", batch["pixel_values"].shape)  # [4, 3, 224, 224]
    print("input_ids:   ", batch["input_ids"].shape)     # [4, 64]
    print("target_ids:  ", batch["target_ids"].shape)    # [4, 64]
