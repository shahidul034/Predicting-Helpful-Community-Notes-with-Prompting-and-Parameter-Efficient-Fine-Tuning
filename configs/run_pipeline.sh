#!/bin/bash
# Full Pipeline Script for Community Notes Effectiveness Prediction
# Usage: ./run_pipeline.sh [stage]
# Stages: all, data, baselines, prompting, finetuning, inference, evaluation

set -e

# GPU Configuration - Use GPU ID 2
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
INPUT_DIR="data/processed"
OUTPUT_DIR="outputs"
MODEL="qwen3-4b"

# Functions
run_data_pipeline() {
    log_info "=== Running Data Pipeline ==="
    
    log_info "Downloading Community Notes data..."
    log_info "Note: If download fails, use --sample-data flag for testing"
    python src/data/download_data.py --output-dir data/raw || {
        log_warn "Download failed. Generating sample data instead..."
        python src/data/download_data.py --output-dir data/raw --sample-data
    }
    
    log_info "Preprocessing data..."
    python src/data/preprocess_data.py \
        --input-dir data/raw \
        --output-dir data/processed
    
    log_info "Running EDA..."
    python src/data/eda.py \
        --input-dir data/processed \
        --output-dir outputs/visualizations
    
    log_info "Data pipeline complete!"
}

run_baselines() {
    log_info "=== Running Baseline Models ==="
    
    log_info "Training TF-IDF + Logistic Regression..."
    python src/baselines/tfidf_baseline.py \
        --input-dir data/processed \
        --output-dir outputs/models
    
    log_info "Training BERT baseline..."
    python src/baselines/bert_baseline.py \
        --input-dir data/processed \
        --output-dir outputs/models \
        --epochs 3
    
    log_info "Baseline training complete!"
}

run_prompting() {
    log_info "=== Running Prompting Experiments ==="
    
    for method in zero-shot few-shot cot; do
        log_info "Running $method prompting..."
        python src/prompting/run_prompting.py \
            --method $method \
            --input-dir data/processed \
            --output-dir outputs/evaluations \
            --max-samples 500
    done
    
    log_info "Prompting experiments complete!"
}

run_finetuning() {
    log_info "=== Running LoRA Fine-tuning ==="
    
    log_info "Fine-tuning $MODEL with LoRA..."
    python src/finetuning/train_lora.py \
        --model $MODEL \
        --input-dir data/processed \
        --output-dir outputs/models \
        --epochs 3 \
        --batch-size 8
    
    log_info "Fine-tuning complete!"
}

run_inference() {
    log_info "=== Running Inference ==="
    
    MODEL_PATH="outputs/models/${MODEL}_lora"
    
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model not found at $MODEL_PATH. Run finetuning first."
        exit 1
    fi
    
    log_info "Running inference on test set..."
    python src/inference/run_inference.py \
        --mode local \
        --model-path $MODEL_PATH \
        --input-dir data/processed \
        --output-dir outputs/evaluations
    
    log_info "Inference complete!"
}

run_evaluation() {
    log_info "=== Running Evaluation ==="
    
    log_info "Comparing all models..."
    python src/evaluation/evaluate.py \
        --results-dir outputs/evaluations \
        --output-dir outputs/evaluations
    
    log_info "Running ablation study..."
    python src/evaluation/ablation_study.py \
        --input-dir data/processed \
        --output-dir outputs/evaluations
    
    log_info "Running error analysis..."
    python src/evaluation/error_analysis.py \
        --input-dir data/processed \
        --output-dir outputs/evaluations
    
    log_info "Evaluation complete!"
}

print_usage() {
    echo "Usage: $0 [stage]"
    echo ""
    echo "Stages:"
    echo "  data        - Download and preprocess data"
    echo "  baselines   - Train baseline models (TF-IDF, BERT)"
    echo "  prompting   - Run prompting experiments"
    echo "  finetuning  - Fine-tune LLM with LoRA"
    echo "  inference   - Run inference on test set"
    echo "  evaluation  - Evaluate and compare all models"
    echo "  all         - Run complete pipeline"
    echo ""
}

# Main
case "${1:-all}" in
    data)
        run_data_pipeline
        ;;
    baselines)
        run_baselines
        ;;
    prompting)
        run_prompting
        ;;
    finetuning)
        run_finetuning
        ;;
    inference)
        run_inference
        ;;
    evaluation)
        run_evaluation
        ;;
    all)
        log_info "=== Running Complete Pipeline ==="
        run_data_pipeline
        run_baselines
        run_prompting
        run_finetuning
        run_inference
        run_evaluation
        log_info "=== Pipeline Complete ==="
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
