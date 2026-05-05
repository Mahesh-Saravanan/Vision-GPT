"""
Training pipeline for VisionGPT2 image captioning.

Features:
  - Mixed Precision Training (torch.cuda.amp) for 16GB VRAM GPUs.
  - AdamW optimiser with linear warmup + cosine decay.
  - Per-epoch logging of train/val loss and token-level accuracy.
  - Checkpoint saving: best (lowest val loss) and latest.
  - Training curve plot saved at the end.

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import VisionGPT2Model
from dataloader import get_dataloader
from utils import save_checkpoint, plot_training_curves


# ---------------------------------------------------------------------------
# Training config  – edit these before running
# ---------------------------------------------------------------------------
CFG = dict(
    vit_model    = "google/vit-base-patch16-224",
    gpt2_model   = "gpt2",           # swap "gpt2-medium" for 1024-dim decoder
    freeze_vit   = True,             # keep ViT frozen; set False to fine-tune end-to-end
    epochs       = 20,
    batch_size   = 32,               # reduce to 16 if OOM
    max_length   = 64,               # max caption token length
    lr           = 3e-4,
    weight_decay = 0.01,
    warmup_steps = 500,
    grad_clip    = 1.0,
    num_workers  = 4,
    checkpoint_dir = "checkpoints",
    plot_path      = "training_curves.png",
)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device):
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids    = batch["input_ids"].to(device, non_blocking=True)
        target_ids   = batch["target_ids"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(pixel_values, input_ids)           # [B, T, vocab]
            loss   = nn.functional.cross_entropy(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accuracy over non-padding tokens
        with torch.no_grad():
            valid_mask = (target_ids != -100)
            preds      = logits.argmax(dim=-1)
            total_correct += ((preds == target_ids) & valid_mask).sum().item()
            total_tokens  += valid_mask.sum().item()
            total_loss    += loss.item()

        avg_loss = total_loss / (total_tokens / max(target_ids.numel(), 1) + 1e-8)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    avg_acc  = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids    = batch["input_ids"].to(device, non_blocking=True)
        target_ids   = batch["target_ids"].to(device, non_blocking=True)

        with autocast():
            logits = model(pixel_values, input_ids)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
            )

        valid_mask     = (target_ids != -100)
        preds          = logits.argmax(dim=-1)
        total_correct += ((preds == target_ids) & valid_mask).sum().item()
        total_tokens  += valid_mask.sum().item()
        total_loss    += loss.item()

    avg_loss = total_loss / len(loader)
    avg_acc  = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    # Tokeniser (shared between dataloader and model.generate)
    tokenizer = GPT2Tokenizer.from_pretrained(CFG["gpt2_model"])
    tokenizer.pad_token = tokenizer.eos_token

    # Data
    print("\nBuilding dataloaders …")
    train_loader = get_dataloader(
        "train", batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], max_length=CFG["max_length"],
        tokenizer=tokenizer,
    )
    val_loader = get_dataloader(
        "val", batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], max_length=CFG["max_length"],
        tokenizer=tokenizer,
    )

    # Model
    print("\nInitialising VisionGPT2Model …")
    model = VisionGPT2Model(
        vit_model_name=CFG["vit_model"],
        gpt2_model_name=CFG["gpt2_model"],
        freeze_vit=CFG["freeze_vit"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total:,}")

    # Optimiser + scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    )
    total_steps = len(train_loader) * CFG["epochs"]
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CFG["warmup_steps"],
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    # Tracking
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float("inf")

    print("\nStarting training …\n")
    for epoch in range(1, CFG["epochs"] + 1):
        print(f"Epoch {epoch}/{CFG['epochs']}")

        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device)
        v_loss, v_acc = eval_one_epoch(model, val_loader, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        print(f"  Train  loss={t_loss:.4f}  acc={t_acc*100:.2f}%")
        print(f"  Val    loss={v_loss:.4f}  acc={v_acc*100:.2f}%")

        # Always save the latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, v_loss,
            os.path.join(CFG["checkpoint_dir"], "latest.pt"),
        )

        # Save the best checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(
                model, optimizer, epoch, v_loss,
                os.path.join(CFG["checkpoint_dir"], "best.pt"),
            )
            print(f"  ✓ New best saved  (val_loss={v_loss:.4f})")

        print()

    # Plot and save training curves
    plot_training_curves(
        train_losses, val_losses,
        train_accs,   val_accs,
        save_path=CFG["plot_path"],
    )
    print(f"Training complete. Curves saved → {CFG['plot_path']}")


if __name__ == "__main__":
    train()
