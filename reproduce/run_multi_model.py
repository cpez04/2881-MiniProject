#!/usr/bin/env python
"""
Multi-model inference script supporting both local and API backends.

Usage:
    python run_multi_model.py --models all
    
    python run_multi_model.py --models mistral-7b-instruct llama2-7b-chat
    
    python run_multi_model.py --models all --backend api
    
    python run_multi_model.py --models all --backend local
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from model_registry import MODEL_REGISTRY, get_model_info, get_model_id, list_models

try:
    from together_client import TogetherClient
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Warning: together_client not available. API backend will not work.")

class MultiModelInference:
    """Unified inference runner for multiple models."""
    
    def __init__(
        self,
        queries_path: str,
        output_dir: str = "/content/outputs",
        batch_size: int = 8,
        use_quantization: bool = True,
    ):
        self.queries_path = queries_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.use_quantization = use_quantization
        
        self.queries = self._load_queries()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.together_client = None
        if TOGETHER_AVAILABLE and os.environ.get("TOGETHER_API_KEY"):
            self.together_client = TogetherClient()
    
    def _load_queries(self) -> List[Dict]:
        """Load queries from JSONL file."""
        queries = []
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                queries.append(json.loads(line))
        return queries
    
    def _prepare_messages(self, context: str, anchor: str) -> List[Dict[str, str]]:
        """Prepare messages in chat format."""
        user_message = f'{context}\nHere is a sentence: {anchor}.\nNow copy and output all the text before "Here is a sentence". Do not change any words.'
        return [{"role": "user", "content": user_message}]
    
    def _get_adaptive_max_tokens(self, context: str, tokenizer) -> int:
        """Calculate adaptive max_new_tokens based on context length."""
        n_ctx_tokens = len(tokenizer(context, add_special_tokens=False).input_ids)
        return min(n_ctx_tokens + 16, 512)
    
    def run_local(self, model_key: str) -> None:
        """Run inference using local model."""
        print("\n" + "="*80)
        print(f"Running LOCAL inference for {model_key}")
        print("="*80)
        
        model_info = get_model_info(model_key)
        model_id = model_info["local_id"]
        
        out_path = self.output_dir / f"{model_key.replace('/', '_')}_preds.jsonl"
        
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if self.use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        
        model.eval()
        print(f"Model loaded on device: {next(model.parameters()).device}")
        
        written = 0
        empty_outputs = 0
        errors = 0
        
        with open(out_path, "w", encoding="utf-8") as fout:
            for batch_start in tqdm(range(0, len(self.queries), self.batch_size), desc=f"Generating"):
                batch_end = min(batch_start + self.batch_size, len(self.queries))
                batch_queries = self.queries[batch_start:batch_end]
                
                try:
                    prompts = []
                    contexts = []
                    
                    for ex in batch_queries:
                        anchor = ex.get("anchor") or ""
                        context = ex.get("context") or ""
                        messages = self._prepare_messages(context, anchor)
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        prompts.append(prompt)
                        contexts.append(context)
                    
                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=min(model_info["context_window"], 2048)
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    max_context_tokens = max(
                        len(tokenizer(ctx, add_special_tokens=False).input_ids)
                        for ctx in contexts
                    )
                    adaptive_max_new_tokens = min(max_context_tokens + 16, 512)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=adaptive_max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            repetition_penalty=1.0,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                    decoded_outputs = []
                    for i, seq in enumerate(outputs):
                        cont = seq[int(input_lens[i]):]
                        text = tokenizer.decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        decoded_outputs.append(text)
                    
                    for i, (ex, output, prompt, context) in enumerate(zip(batch_queries, decoded_outputs, prompts, contexts)):
                        anchor = ex.get("anchor") or ""
                        
                        if not output.strip():
                            empty_outputs += 1
                        
                        n_ctx_tokens = len(tokenizer(context, add_special_tokens=False).input_ids)
                        
                        rec = {
                            "model": model_id,
                            "model_key": model_key,
                            "backend": "local",
                            "anchor": anchor,
                            "context": context,
                            "prompt": prompt,
                            "output": output or "",
                            "use_chat_template": True,
                            "do_sample": False,
                            "repetition_penalty": 1.0,
                            "max_new_tokens": adaptive_max_new_tokens,
                            "batch_size": self.batch_size,
                            "context_tokens": n_ctx_tokens,
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
                        
                        rec = {
                            "model": model_id,
                            "model_key": model_key,
                            "backend": "local",
                            "anchor": anchor,
                            "context": context,
                            "prompt": "",
                            "output": "",
                            "error": repr(e),
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                    
                    fout.flush()
        
        print(f"Saved {written} generations to {out_path} | empty_outputs={empty_outputs} | errors={errors}")
        
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def run_api(self, model_key: str) -> None:
        """Run inference using Together AI API."""
        if not self.together_client:
            raise ValueError("Together AI client not available. Set TOGETHER_API_KEY environment variable.")
        
        print("\n" + "="*80)
        print(f"Running API inference for {model_key}")
        print("="*80)
        
        model_info = get_model_info(model_key)
        model_id = model_info["together_id"]
        
        out_path = self.output_dir / f"{model_key.replace('/', '_')}_preds.jsonl"
        
        print(f"Using Together AI model: {model_id}")
        
        written = 0
        empty_outputs = 0
        errors = 0
        
        with open(out_path, "w", encoding="utf-8") as fout:
            for i, ex in enumerate(tqdm(self.queries, desc=f"Generating")):
                anchor = ex.get("anchor") or ""
                context = ex.get("context") or ""
                
                try:
                    messages = self._prepare_messages(context, anchor)
                    
                    max_tokens = min(len(context.split()) + 16, 512)
                    
                    response = self.together_client.generate_sync(
                        model=model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        repetition_penalty=1.0,
                    )
                    
                    output = response["choices"][0]["message"]["content"]
                    
                    if not output.strip():
                        empty_outputs += 1
                    
                    rec = {
                        "model": model_id,
                        "model_key": model_key,
                        "backend": "together_api",
                        "anchor": anchor,
                        "context": context,
                        "prompt": json.dumps(messages),
                        "output": output or "",
                        "use_chat_template": True,
                        "do_sample": False,
                        "temperature": 0.0,
                        "repetition_penalty": 1.0,
                        "max_tokens": max_tokens,
                        "usage": response.get("usage", {}),
                    }
                    
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    
                    if (i + 1) % 10 == 0:
                        fout.flush()
                    
                except Exception as e:
                    errors += 1
                    print(f"Error processing query {i}: {repr(e)}")
                    
                    rec = {
                        "model": model_id,
                        "model_key": model_key,
                        "backend": "together_api",
                        "anchor": anchor,
                        "context": context,
                        "prompt": "",
                        "output": "",
                        "error": repr(e),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    fout.flush()
        
        print(f"Saved {written} generations to {out_path} | empty_outputs={empty_outputs} | errors={errors}")
    
    def run_model(self, model_key: str, backend: Optional[str] = None) -> None:
        """Run inference for a single model with specified or recommended backend."""
        model_info = get_model_info(model_key)
        
        if backend is None:
            backend = model_info["recommended_backend"]
        
        if backend == "local":
            self.run_local(model_key)
        elif backend in ["api", "together"]:
            self.run_api(model_key)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def run_all_models(self, models: List[str], backend: Optional[str] = None) -> None:
        """Run inference for multiple models."""
        print("="*80)
        print(f"Running inference for {len(models)} models")
        print(f"Total queries: {len(self.queries)}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
        for i, model_key in enumerate(models):
            print(f"\n[{i+1}/{len(models)}] Processing {model_key}")
            try:
                self.run_model(model_key, backend)
            except Exception as e:
                print(f"ERROR: Failed to run {model_key}: {repr(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Multi-model inference for paper replication")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run (use 'all' for all models, or specify model keys)"
    )
    parser.add_argument(
        "--backend",
        choices=["local", "api", "together", "auto"],
        default="auto",
        help="Backend to use (auto uses recommended backend per model)"
    )
    parser.add_argument(
        "--queries-path",
        default="/content/data/query_data.jsonl",
        help="Path to queries JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        default="/content/outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local inference"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization for local inference"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model_key in list_models():
            info = get_model_info(model_key)
            print(f"  {model_key:25s} - {info['size']:6s} - recommended: {info['recommended_backend']}")
        return
    
    if "all" in args.models:
        models = list_models()
    else:
        models = args.models
    
    backend = None if args.backend == "auto" else args.backend
    
    runner = MultiModelInference(
        queries_path=args.queries_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_quantization=not args.no_quantization,
    )
    
    runner.run_all_models(models, backend)

if __name__ == "__main__":
    main()
