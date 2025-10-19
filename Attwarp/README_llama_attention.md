# Language Model Attention Visualization

This Python script creates attention map visualizations for transformer-based language models using HuggingFace transformers. Now defaults to publicly available models, with Llama support for authenticated users.

## Quick Start (No Authentication Required)

```python
# Install dependencies
!pip install -r requirements.txt

# Run the script (uses DialoGPT-medium by default)
%run llama_attention_visualization.py
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For Llama models (optional - requires authentication):**
   - Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf
   - Install additional package: `pip install huggingface_hub`
   - Authenticate in your notebook:
   ```python
   from llama_attention_visualization import authenticate_huggingface
   authenticate_huggingface()
   ```

## Usage in Jupyter Notebook

### Option 1: Run the Python file directly
```python
%run llama_attention_visualization.py
```

### Option 2: Import and use functions
```python
from llama_attention_visualization import load_llama_model, get_attention_maps, create_attention_heatmap, create_text_attention_overlay

# Load model
model, tokenizer = load_llama_model()

# Your text
text = "Your input text here for attention visualization."

# Get attention maps
attention_weights, tokens, input_ids = get_attention_maps(model, tokenizer, text)

# Create visualizations
heatmap_fig = create_attention_heatmap(tokens, attention_weights)
overlay_fig = create_text_attention_overlay(tokens, attention_weights)
```

## Configuration Options

- **Model**: Choose from these options in the script:
  - **Large models (good for attention visualization):**
    - `"microsoft/DialoGPT-large"` (~1.2GB, good attention patterns)
    - `"microsoft/DialoGPT-medium"` (default, ~800MB)
    - `"gpt2-xl"` (~6GB, very good attention)
  - **Smaller models (for testing):**
    - `"gpt2-medium"` (~1.5GB)
    - `"gpt2"` (~500MB)
    - `"distilgpt2"` (~300MB, fastest)
  - **Llama models (requires authentication):**
    - `"meta-llama/Llama-2-7b-hf"` (~14GB)
    - `"meta-llama/Llama-2-7b-chat-hf"` (~14GB, chat version)

- **Layer and Head**: Modify `LAYER_IDX` and `HEAD_IDX`:
  - `LAYER_IDX = -1` (last layer, default - shows final attention patterns)
  - `LAYER_IDX = 0` (first layer - shows low-level patterns)
  - `HEAD_IDX = 0` (first attention head, default)

- **Input Text**: Change `SAMPLE_TEXT` to your desired input

## Output

The script generates two visualizations:

1. **Attention Heatmap**: Matrix showing attention weights between all token pairs
2. **Text Attention Overlay**: Original text with color-coded attention weights (red = high attention)

## Memory Requirements

- **Llama 7B**: ~14GB VRAM (requires HuggingFace authentication)
- **GPT-2 XL**: ~6GB VRAM
- **DialoGPT Large**: ~1.2GB VRAM
- **DialoGPT Medium**: ~800MB VRAM (default choice)
- **GPT-2**: ~500MB VRAM
- **DistilGPT-2**: ~300MB VRAM (good for testing/low-memory)

For CPU-only or limited GPU memory, use `distilgpt2` or `gpt2` models.

## Troubleshooting

- **Authentication error (401)**:
  - For Llama models: Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
  - Use the `authenticate_huggingface()` function in your notebook
  - Or switch to publicly available models (DialoGPT, GPT-2)

- **CUDA out of memory**:
  - Use smaller models: `distilgpt2`, `gpt2`, `microsoft/DialoGPT-medium`
  - Reduce input text length
  - Use `torch_dtype=torch.float16` (already enabled)

- **Slow loading**:
  - First run downloads models (~300MB to 14GB depending on model)
  - Subsequent runs will be faster
  - Use smaller models for testing

- **Import errors**:
  - Install all requirements: `pip install -r requirements.txt`
  - For Llama auth: `pip install huggingface_hub`

- **Model not found**:
  - Check model name spelling
  - Ensure internet connection for downloads

## Example Output

The script will display:
- Tokenized input with special tokens
- Attention heatmap matrix
- Text with attention weights visualized as colored backgrounds
