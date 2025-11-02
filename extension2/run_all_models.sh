#!/bin/bash
# Run firewall evaluation on all models and generate comparative analysis

set -e  # Exit on error

# Configuration
MIN_OVERLAP=${1:-20}
N_BOOTSTRAP=${2:-1000}
OUTPUT_DIR="results"
FIGURES_DIR="figures"

echo "========================================"
echo "Multi-Model Firewall Evaluation"
echo "========================================"
echo "Min Overlap: $MIN_OVERLAP"
echo "Bootstrap Samples: $N_BOOTSTRAP"
echo "========================================"
echo ""

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $FIGURES_DIR

# Model configurations: (model_name, tokenizer_name)
declare -a MODELS=(
    "Llama-2-7b-chat-hf:meta-llama/Llama-2-7b-chat-hf"
    "Llama-2-13b-chat-hf:meta-llama/Llama-2-13b-chat-hf"
    "Mistral-Instruct-7b:mistralai/Mistral-7B-Instruct-v0.1"
    "Mixtral-8x7B-Instruct-v0.1:mistralai/Mixtral-8x7B-Instruct-v0.1"
    "SOLAR-10.7B-Instruct:upstage/SOLAR-10.7B-Instruct-v1.0"
)

# Run evaluation for each model
for model_config in "${MODELS[@]}"; do
    IFS=':' read -r model_name tokenizer_name <<< "$model_config"

    echo "========================================"
    echo "Evaluating: $model_name"
    echo "========================================"

    # Check if results file exists
    RESULTS_PATH="../results/${model_name}.jsonl"
    if [ ! -f "$RESULTS_PATH" ]; then
        echo "WARNING: Results file not found: $RESULTS_PATH"
        echo "Skipping $model_name"
        echo ""
        continue
    fi

    # Run evaluation
    echo "Running firewall evaluation..."
    python3 firewall_eval.py \
        --model "$model_name" \
        --tokenizer "$tokenizer_name" \
        --min-overlap $MIN_OVERLAP \
        --n-bootstrap $N_BOOTSTRAP \
        --results "$RESULTS_PATH" \
        --contexts "../data/retrieved_contexts.jsonl" \
        --output-dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for $model_name"
        continue
    fi

    # Generate visualizations
    echo "Generating visualizations..."
    python3 visualize.py \
        --results "$OUTPUT_DIR/${model_name}_firewall_results.pkl" \
        --output-dir "$FIGURES_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Visualization failed for $model_name"
    fi

    echo ""
done

echo "========================================"
echo "Generating cross-model comparison..."
echo "========================================"

# Generate cross-model comparison
python3 compare_models.py \
    --results-dir "$OUTPUT_DIR" \
    --output-dir "$FIGURES_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "All evaluations complete!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Figures saved to: $FIGURES_DIR"
    echo "========================================"
else
    echo "ERROR: Cross-model comparison failed"
fi
