#!/usr/bin/env python
# coding: utf-8
"""
Improved LLM inference script with:
1. Chat template for proper formatting
2. Greedy decoding to fix spacing issues
3. Batching for speed
4. Optional 4-bit quantization for memory efficiency
"""

import os
import json
import math
from dataclasses import dataclass
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

queries_path = "/content/data/query_data.jsonl"
os.makedirs("/content/outputs", exist_ok=True)
model_stub = model_name.split("/")[-1].replace(":", "_")
out_path = f"/content/outputs/{model_stub}_preds.jsonl"

gen_cfg = dict(
    max_new_tokens=256,  # Reduced from 512 - adjust based on expected context length
    do_sample=False,     # Greedy decoding for better copying
    temperature=0.0,     # Not used when do_sample=False, but set for clarity
    repetition_penalty=1.0,  # Reduced from 1.8 to avoid breaking spacing
)

BATCH_SIZE = 8  # Adjust based on available VRAM (8-32 recommended)

USE_QUANTIZATION = True  # Set to False if you have enough VRAM or want full precision

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"Using device={device}, dtype={torch_dtype}, writing to {out_path}")
print(f"Batch size: {BATCH_SIZE}, Quantization: {USE_QUANTIZATION}")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

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
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="sdpa",  # Use SDPA for attention
        trust_remote_code=True,
    )

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

model.eval()
print("Model & tokenizer loaded.")
print(f"Model device: {next(model.parameters()).device}")

queries = [json.loads(l) for l in open(queries_path, "r", encoding="utf-8")]
print(f"Loaded {len(queries)} prompts from {queries_path}")

def prepare_prompt_with_chat_template(context, anchor):
    """
    Prepare prompt using the model's chat template for better instruction following.
    """
    user_message = f'{context}\nHere is a sentence: {anchor}.\nNow copy and output all the text before "Here is a sentence". Do not change any words.'
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

def generate_batch(batch_queries):
    """
    Generate outputs for a batch of queries.
    """
    prompts = []
    for ex in batch_queries:
        anchor = ex.get("anchor") or ""
        context = ex.get("context") or ""
        prompt = prepare_prompt_with_chat_template(context, anchor)
        prompts.append(prompt)
    
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048  # Adjust based on model's context window
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["do_sample"],
            temperature=gen_cfg["temperature"] if gen_cfg["do_sample"] else None,
            repetition_penalty=gen_cfg["repetition_penalty"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    
    decoded_outputs = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True  # Helps with punctuation spacing
    )
    
    return decoded_outputs, prompts

written = 0
empty_outputs = 0
errors = 0

with open(out_path, "w", encoding="utf-8") as fout:
    for batch_start in tqdm(range(0, len(queries), BATCH_SIZE), desc=f"Generating with {model_stub}"):
        batch_end = min(batch_start + BATCH_SIZE, len(queries))
        batch_queries = queries[batch_start:batch_end]
        
        try:
            decoded_outputs, prompts = generate_batch(batch_queries)
            
            for i, (ex, output, prompt) in enumerate(zip(batch_queries, decoded_outputs, prompts)):
                anchor = ex.get("anchor") or ""
                context = ex.get("context") or ""
                
                if not output.strip():
                    empty_outputs += 1
                
                rec = {
                    "model": model_name,
                    "anchor": anchor,
                    "context": context,
                    "prompt": prompt,
                    "output": output or "",
                }
                
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            
            fout.flush()
            
        except Exception as e:
            errors += 1
            print(f"Error processing batch {batch_start}-{batch_end}: {repr(e)}")
            
            for ex in batch_queries:
                anchor = ex.get("anchor") or ""
                context = ex.get("context") or ""
                prompt = prepare_prompt_with_chat_template(context, anchor)
                
                rec = {
                    "model": model_name,
                    "anchor": anchor,
                    "context": context,
                    "prompt": prompt,
                    "output": "",
                    "error": repr(e),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            
            fout.flush()

print(f"Saved {written} generations to {out_path} | empty_outputs={empty_outputs} | errors={errors}")

print("\n" + "="*80)
print("Sample outputs:")
print("="*80)
shown = 0
for line in open(out_path, "r", encoding="utf-8"):
    obj = json.loads(line)
    print(f"\nExample {shown + 1}:")
    print("-" * 80)
    print(f"Output (first 500 chars):\n{obj['output'][:500]}")
    shown += 1
    if shown >= 3:
        break
