# Multi-Model Inference Guide

This guide explains how to run inference across all 7 models from the paper using both local and API-based approaches.

## Models Supported

1. **Llama2-Chat-7b** - 7B parameters, recommended: local
2. **Mistral-Instruct-7b** - 7B parameters, recommended: local
3. **SOLAR-10.7b** - 10.7B parameters, recommended: API
4. **Llama2-Chat-13b** - 13B parameters, recommended: API
5. **Vicuna-13b** - 13B parameters, recommended: API
6. **Mixtral-Instruct-8x7b** - 8x7B parameters, recommended: API
7. **WizardLM-13b** - 13B parameters, recommended: API

## Quick Start

### Option 1: Together AI API (Recommended - Easiest)

**Pros:**
- Single API for all 7 models
- Fast inference (no model loading time)
- No VRAM limitations
- Consistent results

**Cons:**
- Requires API key (paid service)
- Costs ~$0.20-0.60 per million tokens

**Setup:**
```bash
# Get API key from https://api.together.xyz/
export TOGETHER_API_KEY="your-api-key-here"

# Install dependencies
pip install aiohttp requests

# Run all models with API
python run_multi_model.py --models all --backend api
```

**Cost Estimate:**
- ~230 queries × 7 models = 1,610 total queries
- Average ~300 tokens per query (input + output)
- Total: ~483,000 tokens ≈ $0.10-0.30 depending on model sizes

### Option 2: Local Inference (Free but Slower)

**Pros:**
- Free (no API costs)
- Full control over generation

**Cons:**
- Requires GPU with sufficient VRAM
- Slower (model loading + generation time)
- May not fit all models on Google Colab

**Setup:**
```bash
# Run smaller models locally (7B models)
python run_multi_model.py --models mistral-7b-instruct llama2-7b-chat --backend local

# Larger models may not fit even with 4-bit quantization on Colab T4
```

### Option 3: Hybrid Approach (Best Balance)

Run small models locally and large models via API:

```bash
# Run with auto backend selection (uses recommended backend per model)
python run_multi_model.py --models all --backend auto
```

This will:
- Run Mistral-7B and Llama2-7B locally (free)
- Run 13B and Mixtral models via API (paid but fast)

## Detailed Usage

### List Available Models

```bash
python run_multi_model.py --list-models
```

Output:
```
Available models:
  llama2-7b-chat            - 7b     - recommended: local
  mistral-7b-instruct       - 7b     - recommended: local
  solar-10.7b               - 13b    - recommended: api
  llama2-13b-chat           - 13b    - recommended: api
  vicuna-13b                - 13b    - recommended: api
  mixtral-8x7b-instruct     - 8x7b   - recommended: api
  wizardlm-13b              - 13b    - recommended: api
```

### Run Specific Models

```bash
# Run only Mistral and Llama2-7B
python run_multi_model.py --models mistral-7b-instruct llama2-7b-chat

# Run only 13B models via API
python run_multi_model.py --models llama2-13b-chat vicuna-13b wizardlm-13b --backend api
```

### Configuration Options

```bash
python run_multi_model.py \
  --models all \
  --backend auto \
  --queries-path /content/data/query_data.jsonl \
  --output-dir /content/outputs \
  --batch-size 8 \
  --no-quantization  # Disable 4-bit quantization (not recommended)
```

## Output Format

Each model generates a JSONL file: `{model_key}_preds.jsonl`

Example output record:
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "model_key": "mistral-7b-instruct",
  "backend": "local",
  "anchor": "What group took home the award...",
  "context": "lead single, \"Far Away\"...",
  "prompt": "[INST] lead single, \"Far Away\"... [/INST]",
  "output": "lead single, \"Far Away\"...",
  "use_chat_template": true,
  "do_sample": false,
  "repetition_penalty": 1.0,
  "max_new_tokens": 256,
  "batch_size": 8,
  "context_tokens": 245
}
```

## Google Colab Usage

### Setup in Colab

```python
# Install dependencies
!pip install transformers accelerate torch sentencepiece bitsandbytes aiohttp requests

# Clone repository
!git clone https://github.com/cpez04/2881-MiniProject.git
%cd 2881-MiniProject/reproduce

# Set API key (if using Together AI)
import os
os.environ["TOGETHER_API_KEY"] = "your-api-key-here"

# Upload your data to /content/data/query_data.jsonl
```

### Run Inference

```python
# Option 1: Run all models with API (fastest, requires API key)
!python run_multi_model.py --models all --backend api

# Option 2: Run small models locally (free but slower)
!python run_multi_model.py --models mistral-7b-instruct llama2-7b-chat --backend local

# Option 3: Hybrid (recommended)
!python run_multi_model.py --models all --backend auto
```

### Monitor Progress

```python
# Check output files
!ls -lh /content/outputs/

# View sample outputs
import json
with open('/content/outputs/mistral-7b-instruct_preds.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        obj = json.loads(line)
        print(f"Output {i+1}: {obj['output'][:200]}...")
```

## Troubleshooting

### "TOGETHER_API_KEY not found"

Set your API key:
```python
import os
os.environ["TOGETHER_API_KEY"] = "your-api-key-here"
```

Get an API key from: https://api.together.xyz/

### "CUDA out of memory" (Local Inference)

Solutions:
1. Reduce batch size: `--batch-size 4` or `--batch-size 2`
2. Use 4-bit quantization (enabled by default)
3. Use API instead: `--backend api`
4. Run smaller models only

### "Model not found" or License Issues

Some models require accepting licenses on HuggingFace:
1. Go to the model page on HuggingFace
2. Accept the license agreement
3. Login in Colab: `huggingface-cli login`

Required for:
- Llama2 models (llama2-7b-chat, llama2-13b-chat)
- Vicuna-13b (based on Llama2)
- WizardLM-13b (based on Llama2)

### Spacing Issues in Output

The v2 code fixes the spacing degradation bug. Verify you're using the latest version:
- Check for startup banner: "IMPROVED LLM INFERENCE CODE - v2"
- Check output has `"use_chat_template": true`
- Check `"backend"` field shows correct backend

### API Rate Limits

If you hit rate limits with Together AI:
1. Reduce concurrency (modify `max_concurrent` in together_client.py)
2. Add delays between requests
3. Use local inference for some models

## Performance Comparison

### Local Inference (Google Colab T4)

| Model | Batch Size | Time per Query | Total Time (230 queries) |
|-------|------------|----------------|--------------------------|
| Mistral-7B | 8 | ~2-3s | ~10-15 min |
| Llama2-7B | 8 | ~2-3s | ~10-15 min |
| Llama2-13B | 4 | ~5-7s | ~25-35 min |
| Mixtral-8x7B | 2 | ~10-15s | ~45-60 min |

**Note:** Larger models may not fit on Colab T4 even with 4-bit quantization.

### API Inference (Together AI)

| Model | Time per Query | Total Time (230 queries) |
|-------|----------------|--------------------------|
| All models | ~0.5-2s | ~2-8 min per model |

**Total time for all 7 models:** ~15-30 minutes

## Best Practices

1. **Save progress incrementally**: Output is written to JSONL after each batch/query, so you can resume if interrupted

2. **Test first**: Run on 5-10 queries before running the full dataset
   ```bash
   # Create a small test file
   head -10 /content/data/query_data.jsonl > /content/data/test_queries.jsonl
   
   # Test
   python run_multi_model.py --models mistral-7b-instruct --queries-path /content/data/test_queries.jsonl
   ```

3. **Monitor costs**: If using API, check your usage at https://api.together.xyz/

4. **Use recommended backends**: The `--backend auto` option uses the recommended backend for each model based on size and VRAM requirements

5. **Keep Colab alive**: For long-running jobs, keep the Colab tab active or use Colab Pro for longer runtimes

## Files

- `run_multi_model.py` - Main inference script
- `model_registry.py` - Model configurations and mappings
- `together_client.py` - Together AI API client
- `llm_improved.py` - Single-model local inference (v2 with bug fixes)
- `llm_improved.ipynb` - Notebook version for single model

## Support

If you encounter issues:
1. Check the output JSONL for error messages
2. Verify your API key is set correctly
3. Ensure you have accepted model licenses on HuggingFace
4. Try reducing batch size for local inference
5. Check the GitHub PR for updates: https://github.com/cpez04/2881-MiniProject/pull/1
