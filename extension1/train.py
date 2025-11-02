#!/usr/bin/env python3
"""
QLoRA-safe fine-tuning script to train a model to refuse copying verbatim text.

Key behavior:
- If --load_in_4bit is set (QLoRA), we SAVE ADAPTERS ONLY and DO NOT MERGE.
  Upload will contain adapter_model.safetensors + adapter_config.json (+ tokenizer files).
- If --load_in_4bit is NOT set, we merge LoRA into the base and save a full model.
  Upload will contain model.safetensors (or shards) + config + tokenizer files.

Other improvements:
- Safe serialization (safetensors) and controlled shard size.
- Robust Hub upload (creates repo if missing).
- Clear prints and sanity checks (non-zero trainable params, non-empty dataset).
"""

import argparse
import json
import os
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login, HfApi, create_repo, whoami
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning to prevent verbatim copying.")

    # Hugging Face authentication
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token (or set HF_TOKEN environment variable)")

    # Dataset selection
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the training JSONL file")

    # Model configuration
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str,
                        default="./finetuned_model",
                        help="Directory to save the fine-tuned model")

    # LoRA hyperparameters
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling parameter)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Which modules to apply LoRA to")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps (unused; we save at end)")

    # Hugging Face Hub options
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model (or adapters) to Hugging Face Hub after training")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for uploading to Hugging Face Hub (e.g., user/repo)")

    # Other options
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA). Adapters will be saved; no merge.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (for testing)")

    return parser.parse_args()


def setup_huggingface_auth(args):
    token = args.hf_token or os.getenv("HF_TOKEN")
    if token:
        print("🔐 Logging in to Hugging Face...")
        login(token=token)
        try:
            me = whoami()
            print(f"✅ Authenticated as: {me.get('name') or me.get('email') or me.get('id')}")
        except Exception:
            pass
    else:
        print("⚠️ No HF token provided. Uploading to Hub will fail if not already logged in.")


def load_jsonl_data(file_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line {i+1}: {e}")
                continue
    return data


def format_conversation(example: Dict[str, Any], tokenizer) -> str:
    if "messages" not in example:
        raise ValueError(f"Expected 'messages' key, got: {list(example.keys())}")

    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )


def prepare_datasets(args, tokenizer):
    print(f"📂 Loading {args.data_file}...")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Dataset file not found: {args.data_file}")

    data = load_jsonl_data(args.data_file, max_samples=args.max_samples)
    print(f"   Loaded {len(data)} examples.")

    formatted_texts = []
    skipped = 0
    for i, example in enumerate(data):
        try:
            formatted = format_conversation(example, tokenizer)
            if formatted.strip():
                formatted_texts.append(formatted)
            else:
                skipped += 1
        except Exception as e:
            print(f"Warning: Skipping example {i}: {e}")
            skipped += 1

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} examples due to formatting issues")

    if len(formatted_texts) == 0:
        raise RuntimeError("No valid training examples after formatting. Check your dataset schema.")

    print(f"✅ Successfully formatted {len(formatted_texts)} examples")

    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=args.max_length,
            return_tensors=None
        )

    print("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def setup_lora_model(model, args):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print("🎯 LoRA Configuration:")
    model.print_trainable_parameters()

    # Sanity check trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found after applying LoRA.")
    return model


def save_adapters_only(trainer, tokenizer, out_dir):
    """
    Save only the LoRA adapters (QLoRA scenario). Also save tokenizer to the same directory.
    """
    adapters_dir = out_dir  # save directly to output_dir
    os.makedirs(adapters_dir, exist_ok=True)
    print(f"💾 Saving LoRA adapters to {adapters_dir}")
    trainer.model.save_pretrained(adapters_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapters_dir)
    print("✅ Saved adapters + tokenizer.")


def save_merged_full_model(trainer, tokenizer, out_dir):
    """
    Merge LoRA into a non-quantized base, then save a full model.
    """
    print("🔀 Merging LoRA adapters with base model...")
    merged = trainer.model.merge_and_unload()

    # Ensure config is coherent
    if hasattr(merged, "config") and merged.config is not None:
        if "llama" in str(merged.config.model_type or "").lower() or "llama" in merged.__class__.__name__.lower():
            merged.config.model_type = "llama"
        elif "llama" in merged.__class__.__name__.lower():
            merged.config.model_type = "llama"

        if merged.config.torch_dtype is None:
            merged.config.torch_dtype = torch.bfloat16

    print(f"💾 Saving merged model to {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(out_dir)
    print("✅ Saved merged full model + tokenizer.")


def push_folder_to_hub(folder_path: str, repo_id: str):
    print(f"🚀 Preparing to push '{folder_path}' to Hub as {repo_id}")
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload from {folder_path}",
        # keep it broad; adapters or merged will both be covered
        allow_patterns=[
            "*.json", "*.safetensors*", "*.model", "tokenizer.*", "generation_config.json"
        ],
    )
    print(f"✅ Successfully uploaded to https://huggingface.co/{repo_id}")


def main():
    args = parse_args()

    print("🚀 Starting LoRA/QLoRA fine-tuning...")
    setup_huggingface_auth(args)
    torch.manual_seed(args.seed)

    if args.use_wandb:
        wandb.init(project="no-copying-finetuning", config=vars(args))

    print("🤖 Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        print("🔧 Using 4-bit quantization (QLoRA)...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Optional: disable cache to avoid warnings with gradient checkpointing (not strictly required here)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    model = setup_lora_model(base_model, args)
    model.train()

    train_dataset = prepare_datasets(args, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_checkpoints",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_strategy="no",
        report_to="wandb" if args.use_wandb else None,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("🎯 Starting training...")
    trainer.train()
    print("✅ Training finished.")

    # SAVE PATHS
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_in_4bit:
        # QLoRA path: save adapters only (no merge here!)
        print("⚠️ QLoRA detected: saving adapters only; skipping merge.")
        save_adapters_only(trainer, tokenizer, args.output_dir)
    else:
        # Full precision path: merge and save a full model
        save_merged_full_model(trainer, tokenizer, args.output_dir)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set.")
        push_folder_to_hub(args.output_dir, args.hub_model_id)

    if args.use_wandb:
        wandb.finish()

    print("✅ All done!")


if __name__ == "__main__":
    main()
