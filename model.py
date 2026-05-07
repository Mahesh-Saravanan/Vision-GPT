"""
VisionGPT2  —  full architecture in pure PyTorch; no HuggingFace model classes.

Every layer (patch embedding, multi-head attention, MLP, positional embedding,
cross-attention, causal masking) is defined explicitly below.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  ViTEncoder                                                  │
  │   PatchEmbedding → CLS token → PositionalEmbedding          │
  │   → ViTBlock × 12  (MSA + MLP, pre-norm)                    │
  │   → LayerNorm → [B, 197, 768]                               │
  └────────────────────────┬─────────────────────────────────────┘
                           │
               LinearAdapter (768 → gpt2_dim)
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │  GPT2Decoder                                                 │
  │   TokenEmb + PosEmb                                         │
  │   → DecoderBlock × 12                                       │
  │       ├─ LN + CausalSelfAttention  + residual               │
  │       ├─ LN + CrossAttention       + residual  ← image      │
  │       └─ LN + MLP                  + residual               │
  │   → LayerNorm → LM Head → [B, T, vocab_size]                │
  └──────────────────────────────────────────────────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  SECTION 1 · Vision Transformer Encoder
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Splits a (C, H, W) image into non-overlapping (P, P) patches and
    projects each flattened patch to embed_dim via a single Conv2d.

    Conv2d with kernel=stride=patch_size is mathematically identical to
    splitting into patches and applying a shared linear projection, but runs
    more efficiently on modern hardware.

    For image_size=224, patch_size=16:
        num_patches = (224 // 16)^2 = 196
        output shape: [B, 196, embed_dim]
    """

    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        embed_dim:   int = 768,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)          # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)          # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)     # [B, num_patches, embed_dim]
        return x


class ViTAttention(nn.Module):
    """
    Multi-head self-attention for the ViT encoder (fully bidirectional;
    every patch attends to every other patch).

    Uses a single fused QKV projection for efficiency, then splits
    into Q, K, V before computing scaled dot-product attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D    = self.num_heads, self.head_dim

        # Fused QKV → split heads
        qkv = self.qkv(x)                              # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                        # each [B, H, N, D]

        # Scaled dot-product attention (no mask — bidirectional)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        return self.proj(x)


class ViTMLP(nn.Module):
    """
    Point-wise feed-forward network used inside ViT blocks.
    Linear → GELU → Linear, with optional dropout.
    Default expansion factor of 4 gives intermediate_dim = 4 × embed_dim.
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1  = nn.Linear(embed_dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ViTBlock(nn.Module):
    """
    Single Vision Transformer encoder block (pre-norm variant):

        x  ←  x  +  Attention( LayerNorm(x) )
        x  ←  x  +  MLP(       LayerNorm(x) )

    Pre-norm (LN before each sub-layer) is more stable for training from
    scratch than the post-norm used in the original BERT paper.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = ViTAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = ViTMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder.  Default config matches ViT-Base/16:
        image_size=224, patch_size=16, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4.

    Steps:
      1. PatchEmbedding   → 196 patch vectors
      2. Prepend [CLS]    → sequence length 197
      3. Add learnable 1D positional embeddings
      4. 12 × ViTBlock
      5. Final LayerNorm

    Output: [B, 197, embed_dim]  (CLS token at index 0)
    """

    def __init__(
        self,
        image_size:  int   = 224,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        embed_dim:   int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches  # 196

        # [CLS] token — learns a global image representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 1D positional embeddings for all 197 positions
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]

        x   = self.patch_embed(pixel_values)               # [B, 196, D]
        cls = self.cls_token.expand(B, -1, -1)             # [B,   1, D]
        x   = torch.cat([cls, x], dim=1)                   # [B, 197, D]
        x   = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)                                # [B, 197, D]


# ============================================================
#  SECTION 2 · Cross-Attention  (the encoder–decoder bridge)
# ============================================================

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: the bridge that injects visual context from
    the ViT encoder into each GPT-2 decoder layer.

        Query  = decoder hidden states    [B, T, C]   — text tokens
        Key    = ViT encoder output       [B, 197, C] — image patches
        Value  = ViT encoder output       [B, 197, C] — image patches

    At every decoder layer every text token attends to all 197 image patches,
    pulling relevant visual context needed to predict the next caption word.
    Separate Q / K / V projections are used (no fused QKV) because the
    query and key/value sequences have different lengths.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        decoder_hidden:  torch.Tensor,   # [B, T, C]
        encoder_output:  torch.Tensor,   # [B, 197, C]
    ) -> torch.Tensor:
        B, T, C = decoder_hidden.shape
        S       = encoder_output.shape[1]
        H, D    = self.num_heads, self.head_dim

        Q = self.q_proj(decoder_hidden).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        K = self.k_proj(encoder_output).view(B, S, H, D).transpose(1, 2)  # [B,H,S,D]
        V = self.v_proj(encoder_output).view(B, S, H, D).transpose(1, 2)  # [B,H,S,D]

        attn = (Q @ K.transpose(-2, -1)) * self.scale   # [B, H, T, S]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, T, C)   # [B, T, C]
        return self.resid_drop(self.out_proj(out))


# ============================================================
#  SECTION 3 · GPT-2 Style Decoder
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention for autoregressive decoding.

    A lower-triangular boolean mask (registered as a buffer so it follows
    the model to GPU) ensures that position i can only attend to positions
    0 … i.  This prevents the decoder from "seeing the future" during
    teacher-forced training and matches the token-by-token generation at
    inference time.
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        max_seq_len: int   = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv        = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out        = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask: 1 = allowed, 0 = blocked (future)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                             # each [B, H, T, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, H, T, T]
        # Mask out future positions by setting their logits to -inf
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.out(x))


class DecoderMLP(nn.Module):
    """Feed-forward network inside each decoder block (same structure as ViTMLP)."""

    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1  = nn.Linear(hidden_dim, inner)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(inner, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DecoderBlock(nn.Module):
    """
    Single GPT-2 decoder block, augmented with cross-attention.

    Pre-norm order (LayerNorm before each sub-layer):

      1. LN + CausalSelfAttention + residual
             ↑ text tokens attend only to past tokens (causal mask)

      2. LN + CrossAttention + residual
             ↑ text tokens attend to ALL ViT image patches (no mask needed)

      3. LN + MLP + residual
             ↑ per-position feed-forward transformation
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        max_seq_len: int   = 256,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.ln1        = nn.LayerNorm(hidden_dim)
        self.self_attn  = CausalSelfAttention(hidden_dim, num_heads, max_seq_len, dropout)
        self.ln_cross   = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.ln2        = nn.LayerNorm(hidden_dim)
        self.mlp        = DecoderMLP(hidden_dim, mlp_ratio, dropout)

    def forward(
        self,
        x:              torch.Tensor,   # [B, T, hidden_dim]
        encoder_output: torch.Tensor,   # [B, 197, hidden_dim]
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln_cross(x), encoder_output)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Decoder(nn.Module):
    """
    Autoregressive transformer decoder (GPT-2 architecture) with cross-attention.

    Components:
      - Token embedding table     [vocab_size, hidden_dim]
      - Positional embedding table [max_seq_len, hidden_dim]
      - depth × DecoderBlock
      - Final LayerNorm
      - LM head  [hidden_dim → vocab_size]
        Weight-tied with token embeddings: same matrix is used both to
        embed input tokens and to project hidden states back to logits.
        This halves the largest parameter block (~38 M params for 768-dim)
        and regularises the shared embedding / output space.

    Default config matches GPT-2 Base:
        vocab_size=50257, hidden_dim=768, depth=12, num_heads=12
    """

    def __init__(
        self,
        vocab_size:  int   = 50257,
        hidden_dim:  int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        max_seq_len: int   = 256,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, max_seq_len, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.ln_f    = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying: lm_head and token_emb share the same weight tensor
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, T]
        encoder_output: torch.Tensor,   # [B, 197, hidden_dim]
    ) -> torch.Tensor:
        B, T   = input_ids.shape
        device = input_ids.device

        pos    = torch.arange(T, device=device).unsqueeze(0)          # [1, T]
        hidden = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            hidden = block(hidden, encoder_output)

        return self.lm_head(self.ln_f(hidden))                         # [B, T, vocab]


# ============================================================
#  SECTION 4 · Full VisionGPT2 Model
# ============================================================

class VisionGPT2Model(nn.Module):
    """
    End-to-end image captioning model.

        Image   →  ViTEncoder  →  LinearAdapter  →  GPT2Decoder  →  Logits

    The LinearAdapter is a single nn.Linear that projects the ViT output
    dimension to the GPT-2 hidden dimension.  When both are 768 (the default)
    it still serves as a learned re-projection between the two representation
    spaces rather than a hard-coded identity mapping.

    Default configuration:
        ViT-Base/16  (embed_dim=768, depth=12, heads=12)
        GPT-2 Base   (hidden_dim=768, depth=12, heads=12)
    """

    def __init__(
        self,
        # ViT
        image_size:  int   = 224,
        patch_size:  int   = 16,
        vit_dim:     int   = 768,
        vit_depth:   int   = 12,
        vit_heads:   int   = 12,
        # Decoder
        vocab_size:  int   = 50257,
        gpt2_dim:    int   = 768,
        gpt2_depth:  int   = 12,
        gpt2_heads:  int   = 12,
        max_seq_len: int   = 256,
        # Shared
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            image_size=image_size, patch_size=patch_size,
            embed_dim=vit_dim, depth=vit_depth, num_heads=vit_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        # Adapter: aligns ViT output dim with GPT-2 hidden dim
        self.adapter = (
            nn.Linear(vit_dim, gpt2_dim)
            if vit_dim != gpt2_dim
            else nn.Identity()
        )

        self.decoder = GPT2Decoder(
            vocab_size=vocab_size, hidden_dim=gpt2_dim,
            depth=gpt2_depth, num_heads=gpt2_heads,
            max_seq_len=max_seq_len, mlp_ratio=mlp_ratio, dropout=dropout,
        )

        self.vocab_size = vocab_size

    # ------------------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values [B,3,224,224] → projected patch embeddings [B,197,gpt2_dim]"""
        return self.adapter(self.encoder(pixel_values))

    def forward(
        self,
        pixel_values: torch.Tensor,   # [B, 3, 224, 224]
        input_ids:    torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """Returns logits [B, T, vocab_size]."""
        return self.decoder(input_ids, self.encode_image(pixel_values))

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_greedy(
        self,
        pixel_values:   torch.Tensor,
        tokenizer,
        max_new_tokens: int = 50,
        device:         str = "cpu",
    ) -> str:
        self.eval()
        enc_out   = self.encode_image(pixel_values.to(device))
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

        for step in range(max_new_tokens):
            logits      = self.decoder(input_ids, enc_out)
            next_logits = logits[:, -1, :].clone()
            if step == 0:
                next_logits[:, tokenizer.eos_token_id] = float("-inf")
            next_id   = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(input_ids[0][1:], skip_special_tokens=True)

    @torch.no_grad()
    def generate_beam_search(
        self,
        pixel_values:   torch.Tensor,
        tokenizer,
        beam_size:      int   = 5,
        max_new_tokens: int   = 50,
        length_penalty: float = 1.0,
        device:         str   = "cpu",
    ) -> str:
        self.eval()
        enc_out = self.encode_image(pixel_values.to(device))
        eos_id  = tokenizer.eos_token_id

        beams:     list = [(0.0, [eos_id])]
        completed: list = []

        for step in range(max_new_tokens):
            if not beams:
                break
            candidates = []
            for score, seq in beams:
                if len(seq) > 1 and seq[-1] == eos_id:
                    completed.append((score, seq))
                    continue
                ids      = torch.tensor([seq], device=device)
                logits   = self.decoder(ids, enc_out)
                lp_all   = F.log_softmax(logits[:, -1, :], dim=-1)[0]
                if step == 0:
                    lp_all[eos_id] = float("-inf")
                top_lp, top_id = lp_all.topk(beam_size)
                for lp, tid in zip(top_lp.tolist(), top_id.tolist()):
                    candidates.append((score + lp, seq + [tid]))

            if not candidates:
                break
            candidates.sort(
                key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                reverse=True,
            )
            beams = candidates[:beam_size]
            if all(s[-1] == eos_id for _, s in beams):
                break

        completed.extend(beams)
        completed.sort(
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True,
        )
        return tokenizer.decode(completed[0][1][1:], skip_special_tokens=True)
