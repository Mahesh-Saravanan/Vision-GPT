"""
Shared utility functions for VisionGPT2 image captioning project.

Contents:
  - save_checkpoint / load_checkpoint  – model persistence helpers
  - plot_training_curves               – loss & accuracy visualisation
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    epoch:    int,
    val_loss: float,
    path:     str,
) -> None:
    """
    Save model + optimiser state to disk.

    The checkpoint stores everything needed to resume training or run inference:
      - full model state_dict
      - optimiser state_dict (for resuming training)
      - epoch index and validation loss (for progress tracking)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "epoch":               epoch,
            "val_loss":            val_loss,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model,
    path:      str,
    device:    str  = "cpu",
    optimizer        = None,
) -> tuple:
    """
    Load a checkpoint saved by save_checkpoint().

    Args:
        model     : VisionGPT2Model instance (weights are loaded in-place)
        path      : path to the .pt file
        device    : map location for torch.load
        optimizer : optional; if provided, its state is also restored

    Returns:
        (epoch, val_loss) – training progress metadata
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"Loaded checkpoint '{path}'  (epoch={epoch}, val_loss={val_loss:.4f})")
    return epoch, val_loss


# ---------------------------------------------------------------------------
# Training curve visualisation
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: list,
    val_losses:   list,
    train_accs:   list,
    val_accs:     list,
    save_path:    str  = "training_curves.png",
    show:         bool = False,
) -> None:
    """
    Plot loss and accuracy curves for training and validation, then save to disk.

    Args:
        train_losses : list of per-epoch average train losses
        val_losses   : list of per-epoch average val losses
        train_accs   : list of per-epoch token-level train accuracies (0-1)
        val_accs     : list of per-epoch token-level val accuracies (0-1)
        save_path    : where to write the PNG file
        show         : if True, call plt.show() (useful in notebooks)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VisionGPT2 – Training Curves", fontsize=14, fontweight="bold")

    # ---- Loss panel ----
    ax_loss.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax_loss.plot(epochs, val_losses,   "r-o", markersize=4, label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ---- Accuracy panel ----
    train_pct = [a * 100 for a in train_accs]
    val_pct   = [a * 100 for a in val_accs]
    ax_acc.plot(epochs, train_pct, "b-o", markersize=4, label="Train Acc")
    ax_acc.plot(epochs, val_pct,   "r-o", markersize=4, label="Val Acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Token Accuracy (%)")
    ax_acc.set_title("Token-Level Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)
