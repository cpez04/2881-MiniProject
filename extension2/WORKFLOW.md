# Extension 2: Complete Workflow Guide

## Overview

Extension 2 evaluates firewall-based defenses for RAG data leakage across **all 5 models** from your reproduction:
- Llama-2-7b-chat-hf
- Llama-2-13b-chat-hf
- Mistral-Instruct-7b
- Mixtral-8x7B-Instruct-v0.1
- SOLAR-10.7B-Instruct

This provides comprehensive comparison of:
1. **Cross-model leakage rates** - Which models leak more?
2. **Firewall effectiveness** - Do thresholds generalize across models?
3. **Detector performance** - Which detector works best for each model?

---

## Quick Start (Recommended)

### **Run All Models at Once**

```bash
cd extension2
./run_all_models.sh
```

This will:
1. ✅ Evaluate all 5 models sequentially
2. ✅ Generate per-model visualizations
3. ✅ Create cross-model comparison plots
4. ✅ Output comprehensive summary tables

**Expected runtime**: ~1.5-2.5 hours for all 5 models (depends on hardware)

**Output**:
- `results/`: Individual model results (`.pkl` and `.csv` files)
- `figures/`: Per-model plots + cross-model comparisons
- `figures/cross_model_summary.csv`: Complete comparison table

---

## Single Model Evaluation

If you want to test on just one model first:

```bash
cd extension2
./run_evaluation.sh "Llama-2-7b-chat-hf" "meta-llama/Llama-2-7b-chat-hf"
```

Or for a different model:

```bash
# Llama-2-13b
./run_evaluation.sh "Llama-2-13b-chat-hf" "meta-llama/Llama-2-13b-chat-hf"

# Mistral
./run_evaluation.sh "Mistral-Instruct-7b" "mistralai/Mistral-7B-Instruct-v0.1"

# Mixtral
./run_evaluation.sh "Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-Instruct-v0.1"

# SOLAR
./run_evaluation.sh "SOLAR-10.7B-Instruct" "upstage/SOLAR-10.7B-Instruct-v1.0"
```

---

## Understanding the Results

### **Per-Model Files**

For each model, you'll get:

```
results/
├── Llama-2-7b-chat-hf_firewall_results.pkl     # Full results
├── Llama-2-7b-chat-hf_ROUGE-L.csv              # ROUGE-L metrics
├── Llama-2-7b-chat-hf_MinHash-Jaccard.csv      # Jaccard metrics
└── Llama-2-7b-chat-hf_TF-IDF-Cosine.csv        # Cosine metrics

figures/
├── Llama-2-7b-chat-hf_safety_utility_frontier.png
└── Llama-2-7b-chat-hf_trigger_rates.png
```

### **Cross-Model Comparisons**

After running all models:

```
figures/
├── cross_model_ROUGE-L.png                      # All models, ROUGE-L detector
├── cross_model_MinHash-Jaccard.png              # All models, Jaccard detector
├── cross_model_TF-IDF-Cosine.png                # All models, Cosine detector
├── cross_model_leakage_rates.png                # Leakage rates by model
├── detector_comparison_across_models.png        # Best detector per model
└── cross_model_summary.csv                      # Complete summary table
```

---

## Key Questions Answered

### **1. Which models leak the most?**

Check `cross_model_leakage_rates.png` to see:
- % of outputs with ≥20 token verbatim overlap
- Ranked by leakage susceptibility

**Example interpretation**: If Llama-2-7b has 35% leakage rate and Mixtral has 20%, Llama-2-7b is more vulnerable to data extraction attacks.

---

### **2. Do firewall thresholds generalize?**

Check `cross_model_ROUGE-L.png` (and other detector plots):
- If curves overlap → threshold generalizes well
- If curves diverge → need per-model tuning

**Example interpretation**: If all models achieve BL ≥ 70% at ROUGE-L τ=0.70, this threshold is robust across architectures.

---

### **3. Which detector works best?**

Check `cross_model_summary.csv`:
- Column "Best_Detector" shows which detector achieved highest BL - FPR
- Column "Op_BL" / "Op_FPR" shows performance at target operating point

**Example interpretation**:
```
Model                  Detector            Op_BL    Op_FPR
Llama-2-7b-chat-hf    MinHash-Jaccard     78.5%    9.2%
Mixtral-8x7B          ROUGE-L             82.1%    8.5%
```
MinHash works best for Llama, ROUGE-L for Mixtral.

---

### **4. Can we achieve BL ≥ 70% with FPR ≤ 10%?**

Look for "Op_Threshold" in `cross_model_summary.csv`:
- If "N/A" → No threshold meets both targets
- If numeric → That threshold achieves target

**Example findings**:
- ✅ Models with lower leakage rates are easier to protect
- ❌ Models with high leakage may require accepting higher FPR
- 🔄 Trade-off differs by model architecture

---

## Interpretation Guide

### **Metrics Explained**

| Metric | Definition | Ideal Value |
|--------|-----------|-------------|
| **Leakage Rate** | % of outputs with ≥20 token overlap | Lower is safer (less leakage) |
| **Blocked Leakage (BL)** | % of leaky outputs caught | Higher is better (70%+ target) |
| **False Positive Rate (FPR)** | % of benign outputs blocked | Lower is better (10% max) |
| **Trigger Rate (TR)** | % of all outputs blocked | Depends on tolerance |
| **Balance** | BL - FPR | Higher is better |

### **Safety-Utility Frontier**

The plots show BL vs FPR curves:
- **Upper-left = best** (high safety, low false alarms)
- **Lower-right = poor** (low safety, high false alarms)
- **Green shaded region** = ideal operating zone (BL ≥ 0.7, FPR ≤ 0.1)

---

## Advanced Options

### **Custom Leakage Threshold (K tokens)**

Test sensitivity to the leakage definition:

```bash
# More lenient (K=10)
./run_all_models.sh 10

# More strict (K=30)
./run_all_models.sh 30
```

### **Faster Testing (Fewer Bootstrap Samples)**

For quick validation (less precise CIs):

```bash
# 100 bootstrap samples instead of 1000
./run_all_models.sh 20 100
```

---

## Expected Findings

Based on the design, you should find:

1. **Leakage varies by model**:
   - Smaller models (Llama-2-7b) may leak more
   - Instruction-tuned models may follow "copy" instructions better

2. **Detector trade-offs**:
   - **ROUGE-L**: Best for verbatim copying, order-sensitive
   - **MinHash-Jaccard**: Good balance for n-gram overlap
   - **TF-IDF Cosine**: Catches paraphrases but higher FPR

3. **Threshold generalization**:
   - Some thresholds work across models
   - Others require per-model tuning

4. **Operating points**:
   - Not all models achieve BL ≥ 0.7 with FPR ≤ 0.1
   - Trade-off depends on model leakage rate

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Script skips a model** | Check if `../results/MODEL.jsonl` exists |
| **"Could not load tokenizer"** | Run `huggingface-cli login` |
| **Out of memory** | Run models individually, or reduce bootstrap samples |
| **Very slow** | Reduce `--n-bootstrap` to 100 for testing |

---

## For Your Report

After running Extension 2, you'll have:

1. ✅ **Quantitative results**: CSV tables with metrics + CIs
2. ✅ **Visualizations**: Safety-utility frontiers for all models
3. ✅ **Cross-model analysis**: Which models are more vulnerable?
4. ✅ **Detector comparison**: Which defense works best?
5. ✅ **Threshold recommendations**: Practical operating points

**Key insights to include**:
- Leakage rate variability across models
- Firewall effectiveness (can achieve targets?)
- Best detector for each model
- Generalizability of thresholds
- Safety-utility trade-offs

---

## Summary

**To run everything:**
```bash
cd extension2
pip install -r requirements.txt
huggingface-cli login  # Enter HF token
./run_all_models.sh
```

**Check results:**
```bash
# View summary table
cat figures/cross_model_summary.csv

# View plots
open figures/cross_model_*.png
```

That's it! You now have comprehensive firewall evaluation across all 5 models.
