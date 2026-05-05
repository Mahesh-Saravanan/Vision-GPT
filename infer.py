"""
Inference script for VisionGPT2 image captioning.

Usage:
    # Greedy decoding (fast)
    python infer.py path/to/image.jpg

    # Beam search (better quality, slower)
    python infer.py path/to/image.jpg --mode beam --beam_size 5

    # Custom checkpoint
    python infer.py path/to/image.jpg --checkpoint checkpoints/best.pt
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer

from model import VisionGPT2Model
from utils import load_checkpoint


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Load a local image, resize to (image_size × image_size), and normalise.

    Returns a float tensor of shape [1, 3, image_size, image_size].
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def load_model(
    checkpoint_path: str,
    vit_model:       str = "google/vit-base-patch16-224",
    gpt2_model:      str = "gpt2",
    device:          str = "cpu",
) -> VisionGPT2Model:
    """Build the model, load checkpoint weights, set to eval mode."""
    model = VisionGPT2Model(
        vit_model_name=vit_model,
        gpt2_model_name=gpt2_model,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def caption_image(
    image_path:      str,
    checkpoint_path: str  = "checkpoints/best.pt",
    mode:            str  = "greedy",   # "greedy" or "beam"
    beam_size:       int  = 5,
    max_new_tokens:  int  = 50,
    length_penalty:  float = 1.0,
    vit_model:       str  = "google/vit-base-patch16-224",
    gpt2_model:      str  = "gpt2",
    device:          str  = None,
) -> str:
    """
    End-to-end caption generation for a single image.

    Args:
        image_path      : path to a local image file (jpg / png / …)
        checkpoint_path : path to the saved model checkpoint
        mode            : 'greedy' for fast decoding, 'beam' for better quality
        beam_size       : number of beams (used only when mode='beam')
        max_new_tokens  : maximum caption length in tokens
        length_penalty  : exponent applied to sequence length in beam scoring
        vit_model       : HuggingFace ViT model name
        gpt2_model      : HuggingFace GPT-2 model name
        device          : 'cuda', 'cpu', or None (auto-detect)

    Returns:
        Generated caption string.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokeniser
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = load_model(checkpoint_path, vit_model, gpt2_model, device)

    # Image
    pixel_values = preprocess_image(image_path).to(device)

    # Generate
    if mode == "beam":
        caption = model.generate_beam_search(
            pixel_values, tokenizer,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            device=device,
            length_penalty=length_penalty,
        )
    else:
        caption = model.generate_greedy(
            pixel_values, tokenizer,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    return caption


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="VisionGPT2 – Image Captioning Inference")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--mode", type=str, default="greedy", choices=["greedy", "beam"],
        help="Decoding strategy: 'greedy' or 'beam' (default: greedy)",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Number of beams for beam search (default: 5)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=50,
        help="Maximum number of new tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--length_penalty", type=float, default=1.0,
        help="Length penalty exponent for beam scoring (default: 1.0)",
    )
    parser.add_argument(
        "--gpt2_model", type=str, default="gpt2",
        help="GPT-2 variant used at training time (default: gpt2)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cuda' or 'cpu' (default: auto-detect)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Image      : {args.image}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Mode       : {args.mode}" + (f"  (beam_size={args.beam_size})" if args.mode == "beam" else ""))

    caption = caption_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        beam_size=args.beam_size,
        max_new_tokens=args.max_tokens,
        length_penalty=args.length_penalty,
        gpt2_model=args.gpt2_model,
        device=args.device,
    )

    print(f"\nGenerated caption:\n  {caption}")
