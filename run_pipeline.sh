#!/bin/bash
# Pipeline Script for Community Notes Effectiveness Prediction
# Automatically sets GPU 2 and runs pipeline stages

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default GPU configuration
export CUDA_VISIBLE_DEVICES="2"
GPU_ID="2"

echo "=== Community Notes Pipeline ==="
echo "Project root: $SCRIPT_DIR"
echo "GPU ID: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

# Print usage
usage() {
    echo "Usage: $0 [STAGE] [OPTIONS]"
    echo ""
    echo "Stages:"
    echo "  data          - Download and preprocess data"
    echo "  baselines     - Run TF-IDF and BERT baselines"
    echo "  prompting     - Run prompting experiments"
    echo "  finetuning    - Fine-tune with LoRA"
    echo "  inference     - Run inference with vLLM"
    echo "  evaluation    - Evaluate all models"
    echo "  all           - Run all stages"
    echo ""
    echo "Options:"
    echo "  --model MODEL       - Model for finetuning (qwen3-4b, gemma-3-4b)"
    echo "  --epochs N          - Training epochs"
    echo "  --batch-size N      - Batch size"
    echo "  --max-samples N     - Max samples for prompting"
    echo ""
    echo "Examples:"
    echo "  $0 data"
    echo "  $0 finetuning --model qwen3-4b --epochs 3"
    echo "  $0 all"
    exit 0
}

# Parse arguments
STAGE=""
MODEL="qwen3-4b"
EPOCHS=3
BATCH_SIZE=8
MAX_SAMPLES=""
BERT_EPOCHS=1
BERT_BATCH_SIZE=32

while [[ $# -gt 0 ]]; do
    case $1 in
        data|baselines|prompting|finetuning|inference|evaluation|all)
            STAGE="$1"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$STAGE" ]; then
    usage
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment."
fi

echo ""

# Run stages
case $STAGE in
    data)
        echo ">>> Stage: Data Preparation"
        echo "--- Downloading data ---"
        python src/data/download_data.py
        echo ""
        echo "--- Preprocessing data ---"
        python src/data/preprocess_data.py
        echo ""
        echo "--- Running EDA ---"
        python src/data/eda.py
        echo ""
        echo ">>> Data preparation complete."
        ;;

    baselines)
        echo ">>> Stage: Baselines"
        echo "--- TF-IDF + Logistic Regression ---"
        python src/baselines/tfidf_baseline.py
        echo ""
        echo "--- BERT Baseline ---"
        python src/baselines/bert_baseline.py --epochs "$BERT_EPOCHS" --batch-size "$BERT_BATCH_SIZE"
        echo ""
        echo ">>> Baselines complete."
        ;;

    prompting)
        echo ">>> Stage: Prompting Experiments"
        MAX_SAMPLES_ARG=""
        if [ -n "$MAX_SAMPLES" ]; then
            MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
        fi

        echo "--- Zero-shot prompting ---"
        python src/prompting/run_prompting.py --method zero-shot $MAX_SAMPLES_ARG
        echo ""

        echo "--- Few-shot prompting ---"
        python src/prompting/run_prompting.py --method few-shot $MAX_SAMPLES_ARG
        echo ""

        echo "--- Chain-of-thought prompting ---"
        python src/prompting/run_prompting.py --method cot $MAX_SAMPLES_ARG
        echo ""
        echo ">>> Prompting experiments complete."
        ;;

    finetuning)
        echo ">>> Stage: Fine-Tuning (LoRA)"
        echo "Model: $MODEL, Epochs: $EPOCHS, Batch size: $BATCH_SIZE"

        if [ "$MODEL" = "all" ]; then
            for model_name in qwen3-4b gemma-3-4b; do
                echo "--- Fine-tuning $model_name ---"
                python src/finetuning/train_lora.py \
                    --model "$model_name" \
                    --epochs "$EPOCHS" \
                    --batch-size "$BATCH_SIZE"
                echo ""
            done
        else
            python src/finetuning/train_lora.py \
                --model "$MODEL" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE"
        fi
        echo ""
        echo ">>> Fine-tuning complete."
        ;;

    inference)
        echo ">>> Stage: Inference"
        echo "--- Running inference ---"
        python src/inference/run_inference.py
        echo ""
        echo ">>> Inference complete."
        ;;

    evaluation)
        echo ">>> Stage: Evaluation"
        echo "--- Evaluating all models ---"
        python src/evaluation/evaluate.py
        echo ""
        echo "--- Ablation study ---"
        python src/evaluation/ablation_study.py
        echo ""
        echo "--- Error analysis ---"
        python src/evaluation/error_analysis.py
        echo ""
        echo ">>> Evaluation complete."
        ;;

    all)
        echo ">>> Running ALL stages"
        echo ""

        bash "$0" data
        bash "$0" baselines
        bash "$0" prompting --max-samples "${MAX_SAMPLES}"
        bash "$0" finetuning --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
        bash "$0" inference
        bash "$0" evaluation

        echo ""
        echo ">>> All stages complete!"
        echo ""
        echo "Results are in:"
        echo "  - outputs/models/        (model checkpoints)"
        echo "  - outputs/evaluations/   (evaluation results)"
        echo "  - outputs/visualizations/ (charts)"
        echo "  - outputs/logs/          (training logs)"
        ;;
esac

echo ""
echo "=== Pipeline Complete ==="
