# VisionGPT2 — Architecture & Design Decisions

## Overview

VisionGPT2 is an image captioning system that combines two pretrained transformer models:
a Vision Transformer (ViT) as the visual encoder and GPT-2 as the autoregressive text decoder.
The two models are connected by a lightweight Linear Adapter and a Cross-Attention mechanism
injected into each GPT-2 decoder block.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VisionGPT2Model                             │
│                                                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │               ViT Encoder                │                       │
│  │  google/vit-base-patch16-224 (frozen)    │                       │
│  │                                          │                       │
│  │  224×224 image → 16×16 patches           │                       │
│  │  → 196 patch tokens + 1 CLS token        │                       │
│  │  → last_hidden_state [B, 197, 768]       │                       │
│  └────────────────────┬─────────────────────┘                       │
│                       │                                             │
│          ┌────────────▼────────────┐                                │
│          │  encoder_projection     │  Linear(768 → gpt2_hidden)     │
│          │  (Linear Adapter)       │  learned re-projection         │
│          └────────────┬────────────┘                                │
│                       │ [B, 197, gpt2_hidden]                       │
│                       │                                             │
│  ┌────────────────────│──────────────────────────────────────────┐  │
│  │         Modified GPT-2 Decoder                               │  │
│  │                    │                                          │  │
│  │  Caption tokens → Token + Positional Embeddings              │  │
│  │                    │                                          │  │
│  │  ┌─────────────────▼──────────────────────────────────────┐  │  │
│  │  │  ModifiedGPT2Block  ×  N  (N=12 for gpt2-base)         │  │  │
│  │  │                                                         │  │  │
│  │  │  1. LayerNorm + Causal Self-Attention + Residual        │  │  │
│  │  │     (text tokens attend only to past positions)         │  │  │
│  │  │                    │                                    │  │  │
│  │  │  2. LayerNorm + Cross-Attention + Residual  ◄─── image  │  │  │
│  │  │     Query  = decoder hidden states                      │  │  │
│  │  │     Key    = projected ViT patch embeddings             │  │  │
│  │  │     Value  = projected ViT patch embeddings             │  │  │
│  │  │                    │                                    │  │  │
│  │  │  3. LayerNorm + MLP + Residual                          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                    │                                          │  │
│  │           Final LayerNorm                                     │  │
│  │                    │                                          │  │
│  │          LM Head  [B, T, vocab_size]                          │  │
│  │          (weights tied to token embeddings)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep-Dive

### 1. ViT Encoder (`google/vit-base-patch16-224`)

**What it does:**
The Vision Transformer divides the input image into a grid of fixed-size patches
(16×16 pixels), linearly projects each patch into a 768-dimensional vector, adds
learnable positional embeddings, and passes them through 12 transformer encoder layers.

**Key output:**
`last_hidden_state` — shape `[batch, 197, 768]`
- 196 patch tokens (14×14 grid of 16×16 patches from a 224×224 image)
- 1 CLS token prepended at position 0
All 197 tokens are passed to the decoder's cross-attention; the CLS token carries
a global image summary while patch tokens carry localised spatial detail.

**Why ViT over a CNN:**
ViT processes the image globally from the first layer via self-attention, capturing
long-range spatial relationships (e.g. relating a dog's face to its tail). CNNs build
this global context slowly through deep stacking of local convolutions. This richer
contextual representation gives the cross-attention mechanism more expressive keys/values
to work with.

**Why frozen:**
ViT is pretrained on ImageNet-21k and ImageNet-1k and already produces excellent general
visual features. Freezing it during early training stabilises the cross-attention learning
and dramatically reduces GPU memory usage and training time. The full model can be
unfrozen for a fine-tuning stage once the cross-attention layers have converged.

---

### 2. Linear Adapter (`encoder_projection`)

**What it does:**
A single `nn.Linear(vit_hidden, gpt2_hidden)` layer projects the ViT output from its
hidden dimension to GPT-2's hidden dimension before the visual tokens enter any
cross-attention computation.

**Dimension table:**

| Model Pair           | ViT dim | GPT-2 dim | Projection shape |
|----------------------|---------|-----------|-----------------|
| vit-base + gpt2      |   768   |    768    |  768 → 768      |
| vit-base + gpt2-medium |  768  |   1024    |  768 → 1024     |
| vit-large + gpt2     |  1024   |    768    | 1024 → 768      |

Even in the 768→768 case, this projection is valuable: it is a learned linear
transformation that re-scales and rotates the ViT feature space into a subspace
that is most useful for the GPT-2 attention mechanism, rather than forcing GPT-2
to adapt directly to ViT's raw feature geometry.

---

### 3. Cross-Attention Mechanism

**What it does:**
Cross-attention is the bridge that lets the text decoder "look at" the image.
At every decoder layer and every text token position, the model computes:

```
Q = W_q · decoder_hidden   [B, T, C]  — "what am I looking for?"
K = W_k · encoder_output   [B, S, C]  — "what visual concepts are available?"
V = W_v · encoder_output   [B, S, C]  — "what information to extract?"

Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

Where `T` is the caption length and `S = 197` (ViT patch count).

**Why multi-head:**
Multiple heads (H=12 for gpt2-base, head_dim=64) allow the model to simultaneously
attend to different image regions for different reasons — one head may track
object identity, another colour, another spatial position.

**Why placed between self-attention and MLP:**
This follows the canonical encoder-decoder transformer architecture (Vaswani et al. 2017):
1. Self-attention integrates context from other text tokens already generated.
2. Cross-attention then enriches that text-context with visual information.
3. The MLP refines the combined representation.

Placing cross-attention after the full MLP (i.e. after each complete GPT-2 block) would
mean the text context and the visual context are mixed only at a coarser level, losing
the fine-grained interaction inside each layer's feature space.

**Weights:**
The Q, K, V projection matrices (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and the
`ln_cross` LayerNorm are randomly initialised and trained from scratch. All other
components within `ModifiedGPT2Block` reuse pretrained GPT-2 weights.

---

### 4. GPT-2 Decoder

**Why GPT-2 over BERT:**
GPT-2 is an autoregressive language model trained to predict the next token given
all previous tokens, making it a natural fit for open-ended caption generation.
BERT is a masked language model that sees the full sequence bidirectionally; it
requires additional adaptation (e.g. masked fine-tuning or a separate LM head) to
generate sequences token by token, adding complexity without benefit here.

**Causal masking:**
GPT-2's built-in `GPT2Attention` module maintains a lower-triangular causal mask
as a registered buffer. This ensures that during training (teacher forcing) and
inference (autoregressive decoding), each position can only attend to itself and
earlier positions — preventing information leakage from future tokens.

**Weight tying:**
The LM head (`nn.Linear(gpt2_hidden, vocab_size)`) has its weight matrix set to the
same underlying tensor as the token embedding table (`wte.weight`). This is standard
practice in language models and provides two benefits:
- Reduces the parameter count by ~39M for gpt2-base (50257 × 768).
- Enforces a shared semantic space between input tokens and output predictions,
  often improving generation quality and training stability.

---

### 5. Training Procedure

**Teacher Forcing:**
During training, the ground-truth caption is shifted to create input/target pairs:
```
Full caption tokens : [w1, w2, w3, w4, EOS, PAD, PAD]
input_ids           : [w1, w2, w3, w4, EOS, PAD]      (all but last)
target_ids          : [w2, w3, w4, EOS, PAD, PAD]      (all but first)
```
The model predicts each token given the image and all preceding tokens.
Loss is computed via cross-entropy only at positions where `target_ids != -100`
(padding positions are masked with `-100`, the default `ignore_index` in PyTorch's
`nn.CrossEntropyLoss`).

**The pad_token = eos_token trick:**
GPT-2's tokeniser has no native padding token. The standard workaround is to use the
EOS token (`<|endoftext|>`, id=50256) as padding. To prevent the model from being
penalised for not predicting EOS at padding positions, we explicitly append one real EOS
to each caption before tokenisation and then mask all subsequent EOS tokens with `-100`.
This way the model learns to emit EOS exactly at the end of a real caption.

**Mixed Precision Training (AMP):**
`torch.cuda.amp.autocast` runs the forward pass in float16 where safe (attention,
linear layers) and float32 where numerical precision matters (softmax, layer norm).
`GradScaler` rescales gradients to prevent float16 underflow. Together these reduce
VRAM consumption by ~40-50% and increase throughput by ~1.5-2× on modern GPUs,
allowing a batch size of 32 on a 16GB GPU.

**Optimiser:**
AdamW with weight decay decouples the L2 regularisation from the gradient update,
which is more principled than standard Adam with L2 loss. A cosine learning rate
schedule with linear warmup is used: the warmup phase prevents large destructive
updates to the pretrained GPT-2 weights in the first steps.

---

### 6. Inference

**Greedy decoding:**
At each step, the token with the highest logit is selected. Fast and deterministic
but can produce repetitive or locally suboptimal sequences.

**Beam search:**
Maintains a set of `beam_size` candidate sequences. At each step each beam is
expanded by the top-`beam_size` next tokens; only the overall top-`beam_size`
candidates (scored by cumulative log-probability normalised by sequence length)
are kept. This explores a wider portion of the output distribution and generally
produces more fluent captions at the cost of `beam_size`× more forward passes.

**Length penalty:**
Beam scores are normalised by `len(sequence) ^ length_penalty`. Setting
`length_penalty > 1.0` favours longer captions; `< 1.0` favours shorter ones.
The default of `1.0` is linear normalisation (prevents the model from preferring
trivially short sequences that accumulate less negative log-probability).

---

## File Structure

```
ViT-BERT/
├── model.py        — CrossAttention, ModifiedGPT2Block, VisionGPT2Model
├── dataloader.py   — Flickr30kDataset, get_dataloader
├── train.py        — Training loop, AMP, checkpointing, logging
├── infer.py        — CLI inference with greedy and beam search
└── utils.py        — save/load_checkpoint, plot_training_curves
```

---

## Extending the Project

| Goal | Change |
|------|--------|
| Larger decoder | Set `gpt2_model='gpt2-medium'` in `CFG`; projection auto-adjusts to 768→1024 |
| End-to-end fine-tuning | Set `freeze_vit=False` and lower `lr` to `1e-5` |
| Nucleus sampling | Add `top_p` filtering to `generate_greedy` logits before `argmax` |
| CIDEr / BLEU-4 evaluation | Decode all test images, compare against 5 reference captions with `nltk` or `pycocoevalcap` |
| Faster inference | Add KV-cache by threading `layer_past` through `GPT2Attention` calls in `ModifiedGPT2Block.forward` |
