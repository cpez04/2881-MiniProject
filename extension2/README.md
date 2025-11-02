# Extension 1: Firewall-Based Defense Evaluation

This extension evaluates lightweight "firewall" detectors that block RAG outputs deemed too similar to retrieved context. We test three similarity-based detectors and quantify the safety-utility trade-off.

## Overview

### Goal
Measure how well similarity-based detectors can prevent data leakage from RAG systems while minimizing false positives on legitimate outputs.

### Approach
1. **Ground Truth Labeling**: Mark outputs as "leaky" if they contain ≥20 contiguous tokens matching any retrieved chunk
2. **Detector Evaluation**: Test three detectors at various thresholds:
   - **ROUGE-L (F1)**: Longest common subsequence similarity
   - **MinHash-Jaccard**: 8-gram shingle Jaccard similarity
   - **TF-IDF Cosine**: Cosine similarity on TF-IDF vectors
3. **Threshold Sweeping**: For each detector, sweep thresholds and compute:
   - **Trigger Rate (TR)**: % of outputs blocked
   - **Blocked Leakage (BL)**: % of leaky outputs caught (recall)
   - **False Positive Rate (FPR)**: % of non-leaky outputs blocked
4. **Bootstrap CIs**: 1000× bootstrap for 95% confidence intervals
5. **Comparisons**: Pairwise detector comparison using McNemar's test

## Installation

### Dependencies
```bash
pip install transformers torch numpy pandas matplotlib seaborn scikit-learn datasketch tqdm scipy
```

### Hugging Face Token
Some tokenizers require authentication:
```bash
huggingface-cli login
```

## Directory Structure

```
extension1/
├── README.md                 # This file
├── ground_truth.py          # Leakage labeling (≥K token overlap)
├── detectors.py             # Three detector implementations
├── metrics.py               # Metrics computation + bootstrap
├── firewall_eval.py         # Main evaluation pipeline
├── visualize.py             # Plotting and analysis
├── compare_detectors.py     # Pairwise comparisons
├── run_evaluation.sh        # Convenience script
├── results/                 # Output directory (created automatically)
└── figures/                 # Visualization output
```

## Usage

### Quick Start

Run full evaluation for Llama-2-7b:
```bash
cd extension1
./run_evaluation.sh
```

### Custom Configuration

```bash
./run_evaluation.sh "Llama-2-13b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" 20 1000
#                   ^model_name           ^tokenizer                       ^K  ^bootstrap_samples
```

### Step-by-Step

#### 1. Run Evaluation Pipeline
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

This will:
- Load your reproduction results and retrieved contexts
- Label outputs as leaky/non-leaky using exact token overlap
- Compute similarity scores for all 3 detectors
- Sweep thresholds and compute metrics with bootstrap CIs
- Save results to `results/`

#### 2. Generate Visualizations
```bash
python3 visualize.py \
    --results "results/Llama-2-7b-chat-hf_firewall_results.pkl" \
    --output-dir "figures"
```

This generates:
- **Safety-Utility Frontier**: BL vs FPR curves for all detectors
- **Trigger Rate Plot**: TR vs threshold curves
- **Operating Points Table**: Thresholds meeting BL ≥ 0.7, FPR ≤ 0.1

## Output Files

### Results Directory
After running evaluation:
```
results/
├── Llama-2-7b-chat-hf_firewall_results.pkl    # Full results (pickled)
├── Llama-2-7b-chat-hf_ROUGE-L.csv             # ROUGE-L sweep
├── Llama-2-7b-chat-hf_MinHash-Jaccard.csv     # Jaccard sweep
└── Llama-2-7b-chat-hf_TF-IDF-Cosine.csv       # Cosine sweep
```

### CSV Format
Each CSV contains:
- `threshold`: Detection threshold (τ)
- `trigger_rate`: Fraction of outputs blocked
- `leakage_rate`: Fraction of outputs that are leaky
- `blocked_leakage`: Recall on leaks (BL)
- `false_positive_rate`: False positive rate (FPR)
- `*_ci`: Bootstrap 95% confidence intervals

### Figures Directory
```
figures/
├── Llama-2-7b-chat-hf_safety_utility_frontier.png
└── Llama-2-7b-chat-hf_trigger_rates.png
```

## Key Metrics

### Trigger Rate (TR)
**Definition**: Fraction of all outputs that would be blocked
**Formula**: TR = P(sim ≥ τ)
**Interpretation**: Higher TR = more blocking (lower utility if blocking benign outputs)

### Blocked Leakage (BL)
**Definition**: Fraction of leaky outputs caught by the firewall (recall)
**Formula**: BL = P(sim ≥ τ | leaky)
**Interpretation**: Higher BL = better at catching leaks (safety)

### False Positive Rate (FPR)
**Definition**: Fraction of non-leaky outputs incorrectly blocked
**Formula**: FPR = P(sim ≥ τ | not leaky)
**Interpretation**: Lower FPR = fewer false alarms (better utility)

### Trade-off
- **Safety-Utility Frontier**: Plot BL (y) vs FPR (x)
  - Upper-left = ideal (high BL, low FPR)
  - Lower-right = poor (low BL, high FPR)
- **Operating Point**: Threshold achieving BL ≥ 0.7 with FPR ≤ 0.1

## Detectors

### 1. ROUGE-L (F1)
- **Method**: Longest Common Subsequence (LCS) on tokens
- **Thresholds**: τ ∈ {0.50, 0.55, ..., 0.95}
- **Strengths**: Captures word order; standard MT metric
- **Weaknesses**: Sensitive to reordering; may miss paraphrases

### 2. MinHash-Jaccard
- **Method**: 8-gram shingle Jaccard similarity
- **Parameters**: 192 hash functions, 6 bands × 32 rows
- **Thresholds**: τ ∈ {0.60, 0.65, ..., 0.95}
- **Strengths**: Fast; handles local n-gram overlap well
- **Weaknesses**: Fixed n-gram size; brittle to substitutions

### 3. TF-IDF Cosine
- **Method**: Cosine similarity on TF-IDF vectors (unigram + bigram)
- **Thresholds**: τ ∈ {0.40, 0.50, ..., 0.95}
- **Strengths**: Captures semantic similarity; domain-robust
- **Weaknesses**: Misses verbatim copying if words reweighted differently

## Ablation: Min Overlap Sensitivity

Test robustness to leakage definition:
```bash
# K = 10
python3 firewall_eval.py --min-overlap 10 ...

# K = 30
python3 firewall_eval.py --min-overlap 30 ...
```

Expected: Conclusions should hold across reasonable K values (10-30).

## Expected Findings

Based on the design:
1. **ROUGE-L**: Strong for near-verbatim copying; sensitive to order
2. **MinHash-Jaccard**: Favorable BL/FPR trade-off due to shingle sensitivity
3. **TF-IDF Cosine**: Catches paraphrases but higher FPR

With 230 samples, bootstrap CIs will be ~5-10% wide but sufficient for detector comparison.

## Limitations

- **Sample Size**: 230 queries → wide CIs; trends reliable but point estimates noisy
- **Binary Leakage**: K-token threshold is coarse; misses partial leaks < K tokens
- **Tokenization**: HF tokenizer may differ from model's actual tokenizer
- **No Semantic Eval**: Can't distinguish harmful vs. harmless similarity
- **Offline Only**: Simulates deployment; doesn't test runtime latency

## Future Work

1. **Adaptive Thresholds**: Per-query τ based on query sensitivity
2. **Ensemble Detectors**: Combine ROUGE + Jaccard for better BL-FPR
3. **Semantic Similarity**: Add SBERT/embedding cosine (slower but more robust)
4. **Online A/B Test**: Deploy and measure real user impact
5. **Defense Composition**: Test firewall + fine-tuned "won't-copy" model

## Citation

If using this extension, cite the original paper:

```bibtex
@article{qi2024follow,
  title={Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems},
  author={Qi, Zhenting and Zhang, Hanlin and Xing, Eric and Kakade, Sham and Lakkaraju, Himabindu},
  journal={arXiv preprint arXiv:2402.17840},
  year={2024}
}
```

## Contact

For questions or issues, open an issue in the repository.
