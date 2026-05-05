# Role
You are an expert Deep Learning Engineer specializing in Multi-modal AI using PyTorch and Hugging Face.

# Objective
Generate a modular, well-documented PyTorch project for an Image Captioning system using the Flickr30k dataset.

# Motto & Architecture
- **Encoder:** Vision Transformer (ViT) (e.g., `google/vit-base-patch16-224`).
- **Decoder:** Pre-trained GPT-2.
- **The Bridge:** Implement a custom `VisionGPT2` class. You must explicitly define a **Cross-Attention** mechanism within the decoder blocks so the GPT-2 layers can attend to the ViT's visual patch embeddings.
- **Adapter:** Include a Linear Projection layer to align the hidden dimensions of the ViT output (e.g., 768) with the GPT-2 input (e.g., 768 or 1024).

# Deliverables
Please generate the following Python scripts:

1. **model.py**: 
   - Define the `CrossAttention` layer.
   - Define a `ModifiedGPT2Block` that integrates this Cross-Attention.
   - Assemble the full `VisionGPT2Model` connecting the ViT encoder to the modified GPT-2 decoder.
2. **dataloader.py**: 
   - Load Flickr30k using the `datasets` library.
   - **Handling 5 Captions:** Flatten the dataset so every image-caption pair is a unique sample for training.
   - **Preprocessing:** Resize images to 224x224 and normalize. 
   - **Tokenizer:** Use `GPT2Tokenizer`. Set the `pad_token` to the `eos_token` to handle batching correctly.
3. **train.py**: 
   - Setup a training loop with `torch.cuda.amp` for Mixed Precision (fitting 16GB VRAM).
   - Implement checkpoint saving and basic logging of loss/accuracy.
4. **infer.py**: 
   - A script to load a saved checkpoint, take a local image path, and generate a caption using greedy search or beam search.
5. **utils.py**: 
   - Include a function to plot training/validation loss curves.

# Technical Constraints
- Use `torch`, `torchvision`, and `transformers`.
- Avoid overly complex abstractions; prioritize clear, readable PyTorch code.
- Ensure the model can be initialized with pre-trained weights from Hugging Face for both ViT and GPT-2.