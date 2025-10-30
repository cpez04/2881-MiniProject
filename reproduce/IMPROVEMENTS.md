# LLM Inference Improvements

## Version History

**v2 (Current)**: Fixed critical per-example slicing bug in batched generation that caused spacing degradation
**v1**: Initial improvements with chat template, greedy decoding, batching, and quantization

## Problem Summary

The original LLM inference code had two main issues:

1. **Missing Spaces in Output**: The generated text had missing spaces between words (e.g., "TimeHonorVoice...,wasreleasedinSeptember" instead of "Time Honor Voice..., was released in September")
2. **Slow Inference Speed**: Processing queries one at a time was inefficient on Google Colab

## Critical Bug Fixed in v2

**Per-Example Slicing Bug**: The initial improved code (v1) had a bug where it used a single `input_len` for the entire batch when slicing continuations. This caused incorrect decoding for padded sequences and led to spacing degradation partway through generation. 

**Symptom**: Output would start with correct spacing but degrade into missing spaces (e.g., "fromhis fifthalbum", "end-ofthe-yeawardshows")

**Fix**: Now uses `attention_mask.sum(dim=1)` to get the correct input length for each example in the batch

## Root Causes

### Missing Spaces Issue

The spacing problem was caused by:

1. **Not using the chat template**: Mistral-7B-Instruct-v0.2 is a chat model trained with a specific format. Using raw text prompts instead of the chat template caused the model to be "off-distribution" and produce malformed outputs.

2. **Aggressive repetition penalty**: The original `repetition_penalty=1.8` was too high for a copying task. This pushed the model to choose unusual tokens, including continuation tokens (without leading spaces) where word-start tokens were expected.

3. **Sampling with low temperature**: The combination of `do_sample=True`, `temperature=0.2`, `top_k=60`, `top_p=0.9`, and high `repetition_penalty` created conditions where the model would choose bizarre token sequences.

### Speed Issue

The original code processed queries one at a time without batching, which is inefficient on GPU hardware.

## Solutions Implemented

### 1. Chat Template Usage

**Before:**
```python
prompt = f"{context}\nHere is a sentence: {anchor}.\nNow copy and output all the text before \"Here is a sentence\". Do not change any words."
```

**After:**
```python
def prepare_prompt_with_chat_template(context, anchor):
    user_message = f'{context}\nHere is a sentence: {anchor}.\nNow copy and output all the text before "Here is a sentence". Do not change any words.'
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt
```

This ensures the model receives prompts in the format it was trained on, improving instruction following and output formatting.

### 2. Greedy Decoding

**Before:**
```python
gen_cfg = dict(
    max_new_tokens=512,
    temperature=0.2,
    do_sample=True,
    top_k=60,
    top_p=0.9,
    num_beams=1,
    repetition_penalty=1.8,
)
```

**After:**
```python
gen_cfg = dict(
    max_new_tokens=256,
    do_sample=False,      # Greedy decoding
    temperature=0.0,
    repetition_penalty=1.0,  # Reduced from 1.8
)
```

Greedy decoding (always choosing the most likely token) is more appropriate for copying tasks and eliminates the spacing issues.

### 3. Batching

**Before:** Processing one query at a time
```python
for ex in queries:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, ...)
    decoded = tokenizer.decode(outputs[0][inp_len:], ...)
```

**After:** Processing multiple queries in batches
```python
def generate_batch(batch_queries):
    prompts = [prepare_prompt_with_chat_template(...) for ex in batch_queries]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, ...)
    decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], ...)
    return decoded_outputs

for batch_start in range(0, len(queries), BATCH_SIZE):
    batch_queries = queries[batch_start:batch_end]
    decoded_outputs, prompts = generate_batch(batch_queries)
```

Batching processes multiple queries simultaneously, significantly improving throughput.

### 4. 4-bit Quantization

**Added:**
```python
if USE_QUANTIZATION and torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
```

4-bit quantization reduces memory usage by ~75% and can speed up inference on memory-bound systems like Google Colab.

### 5. Adaptive max_new_tokens (v2)

**Added:**
```python
max_context_tokens = max(len(tokenizer(ctx, add_special_tokens=False).input_ids) for ctx in contexts)
adaptive_max_new_tokens = min(max_context_tokens + 16, gen_cfg["max_new_tokens"])
```

Caps generation length based on context size to prevent long "tail" hallucinations where spacing tends to degrade.

### 6. Debugging Fields (v2)

**Added to each output record:**
```python
rec = {
    ...
    "use_chat_template": True,
    "do_sample": gen_cfg["do_sample"],
    "repetition_penalty": gen_cfg["repetition_penalty"],
    "max_new_tokens": gen_cfg["max_new_tokens"],
    "batch_size": BATCH_SIZE,
    "context_tokens": n_ctx_tokens,
}
```

These fields help diagnose issues and verify the correct code is running.

### 7. Startup Banner (v2)

**Added:**
```
================================================================================
IMPROVED LLM INFERENCE CODE - v2 (with per-example slicing fix)
================================================================================
```

Helps users verify they're running the improved code, not the original.

### 8. Other Improvements

- **Truncation enabled**: Set `truncation=True` with `max_length=2048` to handle very long contexts efficiently
- **Batch flushing**: Write results after each batch instead of after each query for better I/O performance
- **clean_up_tokenization_spaces=True**: Improves punctuation spacing and readability

## Usage

### Using the Improved Notebook (llm_improved.ipynb)

1. Upload the notebook to Google Colab
2. Upload your data to `/content/data/query_data.jsonl`
3. Run all cells in order
4. Results will be saved to `/content/outputs/Mistral-7B-Instruct-v0.2_preds.jsonl`

### Configuration Options

You can adjust these parameters in Cell 4:

```python
# Batch size: Higher = faster but uses more VRAM
# Recommended: 8-16 for T4, 16-32 for A100/L4
BATCH_SIZE = 8

# Quantization: Set to False if you have enough VRAM
USE_QUANTIZATION = True

# Max tokens to generate per query
gen_cfg["max_new_tokens"] = 256
```

### Expected Performance Improvements

Based on typical Google Colab configurations:

| Configuration | Speed Improvement | Memory Usage |
|--------------|-------------------|--------------|
| Original (no batching, no quantization) | 1x (baseline) | ~14GB |
| Batching only (batch_size=8) | ~3-4x faster | ~14GB |
| Quantization only | ~1.2x faster | ~4GB |
| Batching + Quantization | ~4-5x faster | ~4-6GB |

Actual performance will vary based on:
- GPU type (T4 vs L4 vs A100)
- Context length
- Batch size

## Verification

To verify the spacing issue is fixed, compare outputs:

**Original output (with spacing issues):**
```
TimeHonorVoice...,wasreleasedinSeptember  2002.[It']ssingle," GoodPerson'(적당한인간)"rankednumberoneforfourweeksonSouthKorea['MusicBoxChart']
```

**Improved output (with proper spacing):**
```
Time Honor Voice..., was released in September 2002. [It's] single, "Good Person" (적당한 인간)" ranked number one for four weeks on South Korea's Music Box Chart
```

## Troubleshooting

### If you still see spacing issues:

1. Verify you're using the chat template (check the `prepare_prompt_with_chat_template` function is being called)
2. Confirm `do_sample=False` and `repetition_penalty=1.0`
3. Check that `clean_up_tokenization_spaces=True` in the decode call

### If inference is slow:

1. Increase `BATCH_SIZE` (try 16 or 32 if you have enough VRAM)
2. Enable quantization with `USE_QUANTIZATION = True`
3. Reduce `max_new_tokens` if you don't need long outputs
4. Check GPU utilization with `nvidia-smi` to ensure the GPU is being used

### If you run out of memory:

1. Reduce `BATCH_SIZE` (try 4 or 2)
2. Enable quantization with `USE_QUANTIZATION = True`
3. Reduce `max_length` in the tokenizer call (currently 2048)

## Files

- `llm_improved.ipynb`: Jupyter notebook with all improvements (recommended for Google Colab)
- `llm_improved.py`: Python script version (for command-line usage)
- `llm.ipynb`: Original notebook (kept for reference)

## Additional Notes

- The improved code uses `tokenizer.apply_chat_template()` which automatically formats prompts correctly for Mistral-Instruct models
- Greedy decoding is deterministic, so running the same query twice will produce identical outputs
- If you need stochastic outputs, you can set `do_sample=True` but keep `repetition_penalty=1.0` and use lower `temperature` (0.3-0.5)
