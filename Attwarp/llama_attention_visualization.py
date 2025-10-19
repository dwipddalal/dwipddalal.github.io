#!/usr/bin/env python3
"""
Llama 7B Attention Map Visualization

This script loads a Llama 7B model, processes input text, and creates
a colored attention map visualization overlaid on the original text.

Requirements:
- torch
- transformers
- matplotlib
- numpy
- pillow (PIL)

Usage in Jupyter notebook:
1. Install dependencies: !pip install torch transformers matplotlib numpy pillow
2. Run the cells below
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

def authenticate_huggingface():
    """
    Authenticate with HuggingFace for gated models like Llama.

    Call this before loading gated models.
    """
    try:
        from huggingface_hub import login
        print("Please enter your HuggingFace token (get it from https://huggingface.co/settings/tokens):")
        login()
        print("Authentication successful!")
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"Authentication failed: {e}")

def load_llama_model(model_name="microsoft/DialoGPT-medium"):
    """
    Load language model and tokenizer.

    Args:
        model_name (str): HuggingFace model identifier
                         Default is DialoGPT-medium (publicly available)

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with attention outputs enabled
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_attentions=True,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto"  # Automatically distribute across available GPUs
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer

def get_attention_maps(model, tokenizer, text, layer_idx=-1, head_idx=0):
    """
    Get attention maps for input text.

    Args:
        model: Llama model
        tokenizer: Llama tokenizer
        text (str): Input text
        layer_idx (int): Which layer to extract attention from (-1 for last layer)
        head_idx (int): Which attention head to visualize (0 for first head)

    Returns:
        tuple: (attention_weights, tokens, input_ids)
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)

    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    # Extract attention from specified layer and head
    attentions = outputs.attentions[layer_idx]  # Shape: [batch, heads, seq_len, seq_len]
    attention_weights = attentions[0, head_idx].cpu().numpy()  # Shape: [seq_len, seq_len]

    # Decode tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return attention_weights, tokens, input_ids

def create_attention_heatmap(tokens, attention_weights, figsize=(12, 8)):
    """
    Create a heatmap visualization of attention weights.

    Args:
        tokens (list): List of token strings
        attention_weights (np.ndarray): Attention matrix
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The heatmap figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(attention_weights, cmap='viridis', aspect='equal')

    # Set tick labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')

    # Set title
    ax.set_title(f'Attention Map (Layer {layer_idx}, Head {head_idx})', fontsize=14, pad=20)

    plt.tight_layout()
    return fig

def create_text_attention_overlay(tokens, attention_weights, layer_idx=-1, head_idx=0, figsize=(15, 10)):
    """
    Create a text visualization with attention weights overlaid as colored backgrounds.

    Args:
        tokens (list): List of token strings
        attention_weights (np.ndarray): Attention matrix
        layer_idx (int): Layer index for title
        head_idx (int): Head index for title
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The overlay figure
    """
    # Get attention weights for the last token (most common visualization)
    last_token_attention = attention_weights[-1, :]  # Attention from last token to all previous

    # Normalize attention weights to 0-1 range for coloring
    normalized_attention = (last_token_attention - last_token_attention.min()) / \
                          (last_token_attention.max() - last_token_attention.min())

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Create text with colored backgrounds
    text = tokenizer.convert_tokens_to_string(tokens)
    words = text.split()

    # Reconstruct word-to-token mapping (approximate)
    word_positions = []
    current_pos = 0

    for word in words:
        word_start = text.find(word, current_pos)
        word_end = word_start + len(word)
        word_positions.append((word_start, word_end))
        current_pos = word_end

    # Create a large image for text rendering
    img_width, img_height = 1200, 800
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    try:
        # Try to use a nice font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw text with colored backgrounds
    y_position = 50
    line_height = 35
    max_width = img_width - 100

    # Simple token-based coloring (each token gets its attention weight)
    x_position = 50
    token_idx = 0

    for token in tokens:
        if token in ['<s>', '</s>', '<pad>', '<unk>']:
            token_idx += 1
            continue

        # Get attention weight for this token
        attention_val = normalized_attention[token_idx] if token_idx < len(normalized_attention) else 0

        # Create color based on attention weight (red for high attention)
        color_intensity = int(attention_val * 255)
        bg_color = (255, 255 - color_intensity, 255 - color_intensity)  # White to red gradient

        # Decode token to text
        token_text = tokenizer.decode(tokenizer.convert_tokens_to_ids([token])[0],
                                    skip_special_tokens=True)

        if not token_text.strip():
            token_idx += 1
            continue

        # Get text bounding box
        bbox = draw.textbbox((x_position, y_position), token_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background rectangle
        draw.rectangle([x_position-2, y_position-2,
                       x_position + text_width + 2, y_position + text_height + 2],
                      fill=bg_color)

        # Draw text
        draw.text((x_position, y_position), token_text, fill='black', font=font)

        x_position += text_width + 5

        # Wrap to next line if needed
        if x_position > max_width:
            x_position = 50
            y_position += line_height

        token_idx += 1

    # Convert PIL image to matplotlib
    ax.imshow(np.array(image))

    # Add title
    ax.set_title(f'Text with Attention Overlay (Layer {layer_idx}, Head {head_idx})\n'
                f'Showing attention weights from the last token to all previous tokens',
                fontsize=14, pad=20)

    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Configuration - Choose from these publicly available models:
    # Large models (good for attention visualization):
    # MODEL_NAME = "microsoft/DialoGPT-large"      # ~1.2GB, good attention patterns
    # MODEL_NAME = "microsoft/DialoGPT-medium"     # ~800MB, default choice
    # MODEL_NAME = "gpt2-xl"                       # ~6GB, very good attention
    #
    # Smaller models (for testing):
    # MODEL_NAME = "gpt2-medium"                   # ~1.5GB
    # MODEL_NAME = "gpt2"                          # ~500MB
    # MODEL_NAME = "distilgpt2"                    # ~300MB, fastest
    #
    # If you have access to Llama models (requires HuggingFace login):
    # MODEL_NAME = "meta-llama/Llama-2-7b-hf"      # ~14GB, requires auth

    MODEL_NAME = "microsoft/DialoGPT-medium"  # Publicly available, good for attention viz
    SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. This is a test sentence for attention visualization."

    # Layer and head to visualize (you can experiment with these)
    LAYER_IDX = -1  # Last layer
    HEAD_IDX = 0    # First head

    try:
        # Load model and tokenizer
        model, tokenizer = load_llama_model(MODEL_NAME)

        # Get attention maps
        attention_weights, tokens, input_ids = get_attention_maps(
            model, tokenizer, SAMPLE_TEXT, LAYER_IDX, HEAD_IDX
        )

        print(f"Input text: {SAMPLE_TEXT}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")

        # Create attention heatmap
        print("\nCreating attention heatmap...")
        heatmap_fig = create_attention_heatmap(tokens, attention_weights)
        plt.show()

        # Create text attention overlay
        print("Creating text attention overlay...")
        overlay_fig = create_text_attention_overlay(tokens, attention_weights, LAYER_IDX, HEAD_IDX)
        plt.show()

        print("\nVisualization complete!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. For Llama models: Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("2. Then login: from huggingface_hub import login; login()")
        print("3. Try smaller models first: 'distilgpt2', 'gpt2', 'microsoft/DialoGPT-medium'")
        print("4. Ensure sufficient GPU memory (Llama 7B needs ~14GB VRAM)")
        print("5. Install packages: pip install torch transformers matplotlib numpy pillow")
