# Quick Start Guide

## Prerequisites

1. **Complete the reproduction** - You should have already run the reproduction pipeline and have:
   - `results/*.jsonl` files with model outputs
   - `data/retrieved_contexts.jsonl` with retrieved chunks

2. **Install dependencies**:
```bash
cd extension2
pip install -r requirements.txt
```

3. **Authenticate with Hugging Face** (OPTIONAL - only for Llama models):
```bash
huggingface-cli login
```

**Why?** Only Llama-2 tokenizers require Meta's license. Mistral/Mixtral/SOLAR are fully open.
- Get token: https://huggingface.co/settings/tokens
- Accept license: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- **Skip if** you only want to run open models (see Option 1B)

## Running the Evaluation

### Option 1A: All 5 Models (Recommended, requires HF token)

```bash
cd extension2
./run_all_models.sh
```

Evaluates: Llama-2-7b, Llama-2-13b, Mistral, Mixtral, SOLAR
- **Runtime**: ~1.5-2.5 hours
- **Output**: Cross-model comparison plots + summary table

### Option 1B: Open Models Only (No HF token needed)

```bash
cd extension2
./run_open_models.sh
```

Evaluates: Mistral, Mixtral, SOLAR only
- **Runtime**: ~45-90 minutes
- **No authentication required**

### Option 2: Single Model Test

```bash
cd extension2
./run_evaluation.sh "Mistral-Instruct-7b" "mistralai/Mistral-7B-Instruct-v0.1"
```

This runs one model:
- Min token overlap K=20
- 1000 bootstrap samples
- All three detectors (ROUGE-L, MinHash-Jaccard, TF-IDF Cosine)

**Expected runtime**: ~15-30 minutes depending on hardware

### Option 3: Step-by-Step (Manual)

#### Step 1: Run Evaluation
```bash
python3 firewall_eval.py \
    --model "Llama-2-7b-chat-hf" \
    --tokenizer "meta-llama/Llama-2-7b-chat-hf" \
    --min-overlap 20 \
    --n-bootstrap 1000 \
    --results "../results/Llama-2-7b-chat-hf.jsonl" \
    --contexts "../data/retrieved_contexts.jsonl" \
    --output-dir "results"
```

#### Step 2: Visualize Results
```bash
python3 visualize.py \
    --results "results/Llama-2-7b-chat-hf_firewall_results.pkl" \
    --output-dir "figures"
```

#### Step 3: Analyze Results
```bash
python3 analyze_results.py \
    --results "results/Llama-2-7b-chat-hf_firewall_results.pkl" \
    --bl-threshold 0.7 \
    --fpr-threshold 0.1
```

## Understanding the Output

### Terminal Output

You'll see:
1. **Data loading**: Number of query-output pairs loaded
2. **Leakage labeling**: Percentage of outputs marked as leaky
3. **Similarity computation**: Progress bars for each detector
4. **Threshold sweeping**: Progress for each detector's threshold grid
5. **Summary statistics**: Key metrics and operating points

### Files Created

```
extension1/
├── results/
│   ├── Llama-2-7b-chat-hf_firewall_results.pkl     # Full results
│   ├── Llama-2-7b-chat-hf_ROUGE-L.csv              # ROUGE-L sweep
│   ├── Llama-2-7b-chat-hf_MinHash-Jaccard.csv      # Jaccard sweep
│   └── Llama-2-7b-chat-hf_TF-IDF-Cosine.csv        # Cosine sweep
└── figures/
    ├── Llama-2-7b-chat-hf_safety_utility_frontier.png
    └── Llama-2-7b-chat-hf_trigger_rates.png
```

## Interpreting Results

### Safety-Utility Frontier Plot

- **X-axis**: False Positive Rate (FPR) - lower is better
- **Y-axis**: Blocked Leakage (BL) - higher is better
- **Ideal region**: Upper-left (high BL, low FPR)
- **Target lines**:
  - Red dashed: BL ≥ 0.7 (catching 70%+ of leaks)
  - Orange dashed: FPR ≤ 0.1 (blocking ≤10% of benign outputs)

### What to Look For

1. **Best Detector**: Which curve is most upper-left?
2. **Operating Point**: Can any detector achieve BL ≥ 0.7 with FPR ≤ 0.1?
3. **Trade-offs**: How much does FPR increase to get 90%+ BL?

### Example Findings

```
OPERATING POINTS (BL ≥ 70%, FPR ≤ 10%)
================================================================================
Detector             Threshold    BL        FPR      TR
ROUGE-L              0.70         72.3%     8.1%     45.2%
MinHash-Jaccard      0.75         78.5%     9.2%     52.1%
TF-IDF-Cosine        0.60         71.1%     12.3%    58.7%
```

**Interpretation**:
- MinHash-Jaccard achieves best BL (78.5%) while staying under 10% FPR
- Would block 52% of all outputs to catch 78% of leaks
- ROUGE-L has lowest trigger rate but catches fewer leaks

## Troubleshooting

### Error: "No module named 'transformers'"
```bash
pip install -r requirements.txt
```

### Error: "Could not load tokenizer"
```bash
huggingface-cli login
# Enter your HF token
```

### Error: "File not found: retrieved_contexts.jsonl"
Make sure you've run the reproduction pipeline first:
```bash
cd ../reproduce
# Run bm25.ipynb to generate retrieved_contexts.jsonl
```

### Low bootstrap samples for testing
For faster testing (lower confidence):
```bash
python3 firewall_eval.py --n-bootstrap 100 ...
```

## Next Steps

### 1. Run Ablation (Different K values)
```bash
# K = 10
./run_evaluation.sh "Llama-2-7b-chat-hf" "meta-llama/Llama-2-7b-chat-hf" 10

# K = 30
./run_evaluation.sh "Llama-2-7b-chat-hf" "meta-llama/Llama-2-7b-chat-hf" 30
```

### 2. Compare Multiple Models
```bash
for model in "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf" "Mistral-Instruct-7b"
do
    ./run_evaluation.sh "$model" "meta-llama/Llama-2-7b-chat-hf"
done
```

### 3. Generate Paper Table
```bash
python3 analyze_results.py \
    --results "results/Llama-2-7b-chat-hf_firewall_results.pkl" \
    --export-latex "results/firewall_table.tex"
```

## Questions?

See `README.md` for detailed documentation on:
- Detector algorithms
- Metric definitions
- Implementation details
- Limitations and future work
