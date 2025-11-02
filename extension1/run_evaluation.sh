#!/bin/bash
# Convenience script to run full firewall evaluation pipeline

# Set default values
MODEL=${1:-"Llama-2-7b-chat-hf"}
TOKENIZER=${2:-"meta-llama/Llama-2-7b-chat-hf"}
MIN_OVERLAP=${3:-20}
N_BOOTSTRAP=${4:-1000}

echo "========================================"
echo "Firewall Evaluation Pipeline"
echo "========================================"
echo "Model: $MODEL"
echo "Tokenizer: $TOKENIZER"
echo "Min Overlap: $MIN_OVERLAP"
echo "Bootstrap Samples: $N_BOOTSTRAP"
echo "========================================"
echo ""

# Paths
RESULTS_PATH="../results/${MODEL}.jsonl"
CONTEXTS_PATH="../data/retrieved_contexts.jsonl"
OUTPUT_DIR="results"
FIGURES_DIR="figures"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $FIGURES_DIR

# Step 1: Run evaluation
echo "Step 1: Running firewall evaluation..."
python3 firewall_eval.py \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --min-overlap $MIN_OVERLAP \
    --n-bootstrap $N_BOOTSTRAP \
    --results "$RESULTS_PATH" \
    --contexts "$CONTEXTS_PATH" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo ""
echo "Step 2: Generating visualizations..."
python3 visualize.py \
    --results "$OUTPUT_DIR/${MODEL}_firewall_results.pkl" \
    --output-dir "$FIGURES_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Visualization failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Figures saved to: $FIGURES_DIR"
echo "========================================"
