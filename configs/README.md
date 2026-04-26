# Community Notes Effectiveness Prediction

Predicting the effectiveness of crowd-sourced fact-checking using fine-tuned LLMs on Twitter/X Community Notes.

## Project Overview

This project fine-tunes LLMs to predict whether a Community Note will be rated "Helpful" by analyzing linguistic features and comparing different approaches (baselines, prompting, fine-tuning).

## Folder Structure

```
sc_project/
├── configs/                    # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Cleaned and processed data
├── docs/                       # Documentation (LaTeX source files)
│   ├── project_presentation.tex
│   └── project_proposal_v2.tex
├── notebooks/                  # Jupyter notebooks for exploration
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── evaluations/           # Evaluation results
│   ├── visualizations/        # Charts and plots
│   └── logs/                  # Training logs
├── src/
│   ├── data/
│   │   ├── download_data.py      # Download Community Notes data
│   │   ├── preprocess_data.py    # Data cleaning and filtering
│   │   └── eda.py                # Exploratory data analysis
│   ├── baselines/
│   │   ├── tfidf_baseline.py     # TF-IDF + Logistic Regression
│   │   └── bert_baseline.py      # BERT fine-tuning baseline
│   ├── prompting/
│   │   └── run_prompting.py      # Zero/Few-shot, CoT prompting
│   ├── finetuning/
│   │   └── train_lora.py         # LoRA fine-tuning with Unsloth
│   ├── inference/
│   │   ├── serve_vllm.py         # vLLM serving
│   │   └── run_inference.py      # Run predictions
│   ├── evaluation/
│   │   ├── evaluate.py           # Model evaluation metrics
│   │   ├── ablation_study.py     # Ablation experiments
│   │   └── error_analysis.py     # Error analysis
│   └── utils/
│       ├── config.py             # Configuration loading
│       ├── logging_utils.py      # Logging utilities
│       └── metrics.py            # Custom metrics
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md
```

## Quick Start

### GPU Configuration

This project uses **GPU ID 2** by default (NVIDIA A100 80GB). The environment variables are set automatically by the pipeline script.

```bash
# Verify GPU configuration
./verify_gpu.sh

# Or manually check
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES  # Should show: 2
```

**Note:** When `CUDA_VISIBLE_DEVICES=2` is set, PyTorch sees it as `device:0` (the first available device).

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download raw data from Twitter Community Notes
python src/data/download_data.py

# OR generate sample data for testing (when real data unavailable)
python src/data/download_data.py --sample-data

# Preprocess and clean data
python src/data/preprocess_data.py

# Run EDA (optional)
python src/data/eda.py
```

### Run Baselines

```bash
# TF-IDF + Logistic Regression (fast, ~5 seconds)
python src/baselines/tfidf_baseline.py

# BERT baseline (slower, ~5-10 minutes on GPU)
# Use --epochs 1 and --batch-size 32 for faster testing
python src/baselines/bert_baseline.py --epochs 1 --batch-size 32
```

### Run Prompting Experiments

```bash
# Zero-shot, Few-shot, and CoT prompting
# Note: Requires GPU and model download (~4GB for Qwen3-4B)
python src/prompting/run_prompting.py --method zero-shot --max-samples 100
python src/prompting/run_prompting.py --method few-shot --max-samples 100
python src/prompting/run_prompting.py --method cot --max-samples 100
```

### Fine-Tune with Unsloth

```bash
# Fine-tune Qwen3-4B-Instruct with LoRA
# Note: Uses GPU 2 (set via CUDA_VISIBLE_DEVICES)
python src/finetuning/train_lora.py --model qwen3-4b --epochs 3 --batch-size 8

# Fine-tune Gemma-3-4B-IT with LoRA
python src/finetuning/train_lora.py --model gemma-3-4b --epochs 3 --batch-size 8

# Or use the pipeline script (automatically sets GPU 2)
./run_pipeline.sh finetuning
```

### Run Inference with vLLM

```bash
# Start vLLM server
python src/inference/serve_vllm.py --model-path outputs/models/qwen3-4b-lora

# Run inference on test data
python src/inference/run_inference.py --endpoint http://localhost:8000
```

### Evaluate Models

```bash
# Evaluate all models
python src/evaluation/evaluate.py --results-dir outputs/evaluations

# Ablation study
python src/evaluation/ablation_study.py

# Error analysis
python src/evaluation/error_analysis.py
```

## Configuration

Edit the YAML files in `configs/` to customize:

- `data_config.yaml`: Data paths, filtering parameters
- `model_config.yaml`: Model names, hyperparameters
- `training_config.yaml`: Training parameters, LoRA settings

## Models

| Approach | Model | Description |
|----------|-------|-------------|
| Baseline 1 | TF-IDF + LR | Traditional ML baseline |
| Baseline 2 | BERT-base | Standard transformer baseline |
| Prompting | Qwen3-4B | Zero-shot, Few-shot, CoT |
| Fine-tuning | Qwen3-4B + LoRA | Primary method |
| Fine-tuning | Gemma-3-4B + LoRA | Architecture comparison |

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- AUC-ROC
- AUC-PR

## Team

- **Md Shahidul Salim**: LLM fine-tuning (LoRA via Unsloth), model training, inference with vLLM
- **Rafa Pashkov**: Data preprocessing, EDA, traditional baselines
- **Mridul Madan**: Prompting experiments, evaluation pipeline, analysis & visualization

## References

1. Allen et al. "Birds of a Feather Don't Fact-Check Each Other." CHI 2022.
2. Chuai et al. "Did the Roll-Out of Community Notes Reduce Engagement?" CSCW 2024.
3. Saeed et al. "Crowdsourced Fact-Checking at Twitter." CIKM 2022.
4. Unsloth: Efficient LLM Fine-Tuning. https://github.com/unslothai/unsloth
