"""
VisionGPT2 - Image Captioning with ViT Encoder + Modified GPT-2 Decoder.

Architecture:
  Image -> ViT -> [B, 197, 768] -> Linear Projection -> [B, 197, gpt2_dim]
                                           |
  Caption -> GPT-2 Embeddings -> ModifiedGPT2Block x N
                                  |- Causal Self-Attention (text tokens)
                                  |- Cross-Attention      (image patches) <-- The Bridge
                                  |- MLP
                                           |
                                     LayerNorm -> LM Head -> Logits [B, T, vocab]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, GPT2Model


# ---------------------------------------------------------------------------
# Cross-Attention: the bridge between visual (ViT) and text (GPT-2) worlds
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention.

    The GPT-2 decoder (text tokens) acts as Query.
    The ViT encoder output (image patches) acts as Key and Value.

    This lets each caption token "look at" all image patches and pull
    relevant visual context while generating the next word.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Separate linear projections for Q, K, V
        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, decoder_hidden: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        decoder_hidden : [B, T, C]  – current GPT-2 hidden states (caption tokens)
        encoder_output : [B, S, C]  – ViT patch embeddings (image context)
        Returns        : [B, T, C]  – decoder hidden enriched with visual context
        """
        B, T, C = decoder_hidden.shape
        S = encoder_output.shape[1]
        H, D = self.num_heads, self.head_dim

        # Project to Q / K / V and split into heads
        Q = self.q_proj(decoder_hidden).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        K = self.k_proj(encoder_output).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        V = self.v_proj(encoder_output).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]

        # Scaled dot-product attention over image patches
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, T, S]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values and merge heads
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.resid_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# Modified GPT-2 Block: causal self-attn → cross-attn → MLP
# ---------------------------------------------------------------------------

class ModifiedGPT2Block(nn.Module):
    """
    Standard GPT-2 block augmented with a Cross-Attention layer.

    Order of operations (following the pre-norm convention GPT-2 uses):
      1. LayerNorm + Causal Self-Attention + Residual  (text only)
      2. LayerNorm + Cross-Attention       + Residual  (text queries image)
      3. LayerNorm + MLP                   + Residual  (feature mixing)
    """

    def __init__(self, gpt2_block, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # ---- Pretrained GPT-2 components (weights already loaded) ----
        self.ln_1     = gpt2_block.ln_1   # LayerNorm before self-attention
        self.self_attn = gpt2_block.attn  # Causal self-attention (pretrained)
        self.ln_2     = gpt2_block.ln_2   # LayerNorm before MLP
        self.mlp      = gpt2_block.mlp    # Feed-forward network (pretrained)

        # ---- New Cross-Attention components (randomly initialised) ----
        self.ln_cross  = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)

    def forward(self, hidden_states: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        hidden_states  : [B, T, C]   – caption token representations
        encoder_output : [B, 197, C] – ViT image-patch embeddings (projected)
        """
        # 1. Causal Self-Attention
        # GPT2Attention applies the causal (lower-triangular) mask internally,
        # so no explicit mask needs to be passed here.
        residual = hidden_states
        hidden_states = self.self_attn(self.ln_1(hidden_states))[0] + residual

        # 2. Cross-Attention
        # Each caption token attends to ALL 197 ViT image patches.
        # This is where visual information is injected into the text stream.
        residual = hidden_states
        hidden_states = self.cross_attn(self.ln_cross(hidden_states), encoder_output) + residual

        # 3. MLP
        residual = hidden_states
        hidden_states = self.mlp(self.ln_2(hidden_states)) + residual

        return hidden_states


# ---------------------------------------------------------------------------
# Full VisionGPT2 Model
# ---------------------------------------------------------------------------

class VisionGPT2Model(nn.Module):
    """
    Image Captioning model: ViT encoder → Linear Adapter → Modified GPT-2 decoder.
    """

    def __init__(
        self,
        vit_model_name:  str   = 'google/vit-base-patch16-224',
        gpt2_model_name: str   = 'gpt2',
        dropout:         float = 0.1,
        freeze_vit:      bool  = True,
    ):
        super().__init__()

        # ---- Encoder: ViT ------------------------------------------------
        self.vit = ViTModel.from_pretrained(vit_model_name)
        vit_hidden = self.vit.config.hidden_size  # 768 for vit-base

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # ---- Load pretrained GPT-2 and unpack components -----------------
        gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        cfg  = gpt2.config
        gpt2_hidden = cfg.hidden_size           # 768 for gpt2, 1024 for gpt2-medium
        num_heads   = cfg.num_attention_heads   # 12 for gpt2

        # ---- Adapter: align ViT dim → GPT-2 dim -------------------------
        # For vit-base + gpt2 both are 768, so this is still a learned re-projection.
        # Swap gpt2_model_name='gpt2-medium' and it handles the 768→1024 case.
        self.encoder_projection = nn.Linear(vit_hidden, gpt2_hidden)

        # ---- GPT-2 Decoder components ------------------------------------
        self.token_emb   = gpt2.wte   # [vocab_size, gpt2_hidden]
        self.pos_emb     = gpt2.wpe   # [max_pos, gpt2_hidden]
        self.emb_dropout = gpt2.drop  # embedding dropout

        # Replace each standard block with the cross-attention augmented version
        self.blocks = nn.ModuleList([
            ModifiedGPT2Block(block, gpt2_hidden, num_heads, dropout)
            for block in gpt2.h
        ])

        self.ln_f = gpt2.ln_f  # final layer norm

        # LM head: project hidden → vocabulary logits
        self.lm_head = nn.Linear(gpt2_hidden, cfg.vocab_size, bias=False)
        # Tie LM head weights with token embeddings (standard practice, saves params)
        self.lm_head.weight = self.token_emb.weight

        self.vocab_size  = cfg.vocab_size
        self.gpt2_hidden = gpt2_hidden

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values : [B, 3, 224, 224]
        Returns      : [B, 197, gpt2_hidden]  projected ViT patch embeddings
        """
        # ViT: 196 patch tokens + 1 CLS token = 197 total
        vit_out = self.vit(pixel_values=pixel_values).last_hidden_state  # [B, 197, 768]
        return self.encoder_projection(vit_out)                          # [B, 197, gpt2_hidden]

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids:    torch.Tensor,
    ) -> torch.Tensor:
        """
        pixel_values : [B, 3, 224, 224]
        input_ids    : [B, T]  – tokenised caption (teacher-forcing input)
        Returns      : [B, T, vocab_size]
        """
        B, T = input_ids.shape
        device = input_ids.device

        encoder_out = self.encode_image(pixel_values)  # [B, 197, gpt2_hidden]

        # GPT-2 token + positional embeddings
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        hidden  = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(pos_ids))

        # Pass through modified blocks; each block attends to encoder_out via cross-attention
        for block in self.blocks:
            hidden = block(hidden, encoder_out)

        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)  # [B, T, vocab_size]

    # ------------------------------------------------------------------
    # Greedy decoding (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_greedy(
        self,
        pixel_values:   torch.Tensor,
        tokenizer,
        max_new_tokens: int = 50,
        device:         str = 'cpu',
    ) -> str:
        self.eval()
        pixel_values = pixel_values.to(device)
        encoder_out  = self.encode_image(pixel_values)

        # GPT-2 uses eos_token_id as the de-facto BOS
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

        for _ in range(max_new_tokens):
            T       = input_ids.shape[1]
            pos_ids = torch.arange(T, device=device).unsqueeze(0)
            hidden  = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(pos_ids))
            for block in self.blocks:
                hidden = block(hidden, encoder_out)
            logits    = self.lm_head(self.ln_f(hidden[:, -1, :]))  # [1, vocab]
            next_id   = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Beam search decoding (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_beam_search(
        self,
        pixel_values:   torch.Tensor,
        tokenizer,
        beam_size:      int   = 5,
        max_new_tokens: int   = 50,
        device:         str   = 'cpu',
        length_penalty: float = 1.0,
    ) -> str:
        self.eval()
        pixel_values = pixel_values.to(device)
        encoder_out  = self.encode_image(pixel_values)  # [1, 197, C]

        eos_id = tokenizer.eos_token_id
        # (log_prob, token_id_list)
        beams:     list = [(0.0, [eos_id])]
        completed: list = []

        def _score_seq(seq_ids: list) -> torch.Tensor:
            ids = torch.tensor([seq_ids], device=device)
            T   = ids.shape[1]
            pos = torch.arange(T, device=device).unsqueeze(0)
            h   = self.emb_dropout(self.token_emb(ids) + self.pos_emb(pos))
            for block in self.blocks:
                h = block(h, encoder_out)
            return self.lm_head(self.ln_f(h[:, -1, :]))  # [1, vocab]

        for _ in range(max_new_tokens):
            if not beams:
                break
            candidates = []
            for log_p, seq in beams:
                if len(seq) > 1 and seq[-1] == eos_id:
                    completed.append((log_p, seq))
                    continue
                logits    = _score_seq(seq)
                log_probs = F.log_softmax(logits, dim=-1)[0]
                top_lp, top_id = log_probs.topk(beam_size)
                for lp, tid in zip(top_lp.tolist(), top_id.tolist()):
                    candidates.append((log_p + lp, seq + [tid]))

            if not candidates:
                break

            # Sort by length-normalised score and keep top beam_size
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = candidates[:beam_size]

            if all(s[-1] == eos_id for _, s in beams):
                break

        completed.extend(beams)
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        best_seq = completed[0][1]
        return tokenizer.decode(best_seq, skip_special_tokens=True)
