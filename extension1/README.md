# Extension 1: Finetuning a "Refuse to Copy" Llama‑2 Chat Model

This extension fine‑tunes a Llama‑2‑Chat model (LoRA/QLoRA) to avoid verbatim copying of retrieved context in RAG answers. It includes a synthetic training dataset that teaches the model to decline word‑for‑word repetition requests and to answer paraphrastically instead.

## Overview

- **Goal**: Train a Llama‑2‑Chat model to refuse verbatim copying and still answer helpfully with synthetic data.
- **Method**: LoRA or QLoRA on conversation data with explicit refusal patterns.
- **Dataset**: `anti_copy_rag_qa.jsonl` (JSONL chat format: `messages: [{role, content}, ...]`). See `RefuseToCopy_SyntheticDatasetCreation.ipynb` for how the synthetic pairs were created.

## What’s in this folder

```
extension1/
├── README.md                              # This file (finetuning-focused)
├── train.py                               # LoRA/QLoRA finetuning script
├── anti_copy_rag_qa.jsonl                 # Training dataset (chat JSONL)
├── RefuseToCopy_SyntheticDatasetCreation.ipynb  # Dataset generation notebook
├── requirements.txt                        # Python deps for finetuning
├── atip-ux-cs-2881-no-copying-finetuning-v7.*   # Example results/artifacts
└── QUICK_START.md                          # (Optional) convenience notes
```

## Setup

### Install
```bash
cd extension1
pip install -r requirements.txt
```

Note that the requirements from the original reproduction are also needed here. 

### (Optional) Hugging Face login
Needed if you will push models or need gated tokenizers.
```bash
huggingface-cli login
```

## Data format and creation

Each line in `anti_copy_rag_qa.jsonl` is a conversation for `tokenizer.apply_chat_template`:

```json
{
  "messages": [
    {"role": "user", "content": "... prompt that tries to elicit copying ..."},
    {"role": "assistant", "content": "... refusal to copy + helpful, paraphrased answer ..."}
  ]
}
```

For replicability, it is recommended to use `anti_copy_rag_qa.jsonl`. However, if desired, one can create their a new synthetic dataset using  `RefuseToCopy_SyntheticDatasetCreation.ipynb`.

## Training

The script supports both LoRA and QLoRA:
- QLoRA: pass `--load_in_4bit` (saves adapters only; no merge)
- LoRA (full‑precision base): omit `--load_in_4bit` (merges adapters and saves a full model)

For full testing, use LoRA. For local tests, QLoRA can be faster. In addition, the script uploads models to hugging face, so model IDs should be changed depending on the user. 

### Quick start (QLoRA, adapters only)
```bash
python train.py --model_name "meta-llama/Llama-2-7b-chat-hf" --data_file "anti_copy_rag_qa.jsonl" --output_dir "./cs-2881-no-copying-finetuning" --hub_model_id "atip-ux/cs-2881-no-copying-finetuning" --push_to_hub --num_epochs 5 --learning_rate 2e-4 --seed 42 --load_in_4bit

```

### Full‑precision merge (saves a standalone model)
```bash
python train.py --model_name "meta-llama/Llama-2-7b-chat-hf" --data_file "anti_copy_rag_qa.jsonl" --output_dir "./cs-2881-no-copying-finetuning" --hub_model_id "atip-ux/cs-2881-no-copying-finetuning" --push_to_hub --num_epochs 5 --learning_rate 2e-4 --seed 42
```

## Testing

After the model is uploaded to hugging face, it should be imported to together AI. From there, one can run the `2881-MiniProject/reproduce/together.ipynb` notebook with the model name changed to see results. 