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

This project uses **GPU ID 3** by default (NVIDIA H200). The environment variables are set automatically by the pipeline script.

```bash
# Verify GPU configuration
./verify_gpu.sh

# Or manually check
nvidia-smi -L
```

### Installation

```bash
# Create conda environment
conda create -n sc_project python=3.12 -y
conda activate sc_project

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
# Note: Uses GPU 3 (configured by the pipeline script)
python src/finetuning/train_lora.py --model qwen3-4b --epochs 3 --batch-size 8

# Fine-tune Gemma-3-4B-IT with LoRA
python src/finetuning/train_lora.py --model gemma-3-4b --epochs 3 --batch-size 8

# Or use the pipeline script (automatically sets GPU 3)
./run_pipeline.sh finetuning
```

### Run Inference with vLLM

```bash
# Start vLLM server
python src/inference/serve_vllm.py --model-path outputs/models/qwen3-4b-lora

# Run inference on test data
python src/inference/run_inference.py --endpoint http://localhost:8000
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
| Fine-tuning | Qwen3-4B + LoRA | [Hugging Face](https://huggingface.co/shahidul034/qwen3-4b-instruct-2507-CommunityNotesEffectiveness) |
| Fine-tuning | Gemma-3-4B + LoRA | [Hugging Face](https://huggingface.co/shahidul034/gemma-3-4b-it-CommunityNotesEffectiveness) |

## Dataset Analysis

### Dataset Statistics

| Statistic | Train (754) | Test (246) |
|-----------|:-----------:|:----------:|
| Helpful (\%) | 49.7% | 54.5% |
| Not Helpful (\%) | 50.3% | 45.5% |
| Misinformed intent (\%) | 53.8% | 47.2% |
| Not Misleading intent (\%) | 46.2% | 52.8% |
| Mean word count | 31.4 | 31.7 |
| Median word count | 33 | 33 |
| Mean ratings per note | 50.1 | 50.2 |
| Flesch-Kincaid grade | 11.2 | 11.0 |
| Date range | Jan--Sep 2024 | Oct--Dec 2024 |

### Class Distribution

```
Training Set (754 notes):
  Helpful (1):       375 (49.7%)
  Not Helpful (0):   379 (50.3%)

Test Set (246 notes):
  Helpful (1):       134 (54.5%)
  Not Helpful (0):   112 (45.5%)
```

### Linguistic Profile

| Feature | Helpful Notes | Not Helpful Notes |
|---------|:-------------:|:-----------------:|
| Mean word count | 32.7 | 30.0 |
| Flesch-Kincaid grade | 11.2 | 11.1 |
| Top unigrams | data, claim, about, tweet | claim, about, data, tweet |
| Top bigrams | `not consistent with`, `publicly available data` | `hard to say`, `for certain` |

**Key linguistic patterns:**
- **Helpful notes** use authoritative, evidence-citing phrasing: *"the original tweet's figures are not consistent with publicly available data from authoritative sources"*
- **Not Helpful notes** use hedging language: *"This is complex... hard to say for certain... worth noting that"*
- Readability is similar across classes (~Grade 11), indicating complexity alone does not distinguish effective notes

### Data Quality

- **Temporal split** ensures zero overlap between train/test (1-day gap: Sep 30 → Oct 1)
- **No duplicate note IDs** across splits
- **Statistically consistent** splits: word count (31.4 vs 31.7), ratings (50.1 vs 50.2), readability (11.2 vs 11.0)

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score

## Team & Contributions

- **Md Shahidul Salim**
  - LLM fine-tuning (LoRA via Unsloth)
  - Model training and inference serving (vLLM)
  - Hugging Face model publishing

- **Rafa Pashkov**
  - Data preprocessing and EDA
  - Implementation of traditional baselines (TF-IDF + LR, BERT)

- **Mridul Madan**
  - Prompting experiments (Zero-shot, Few-shot, CoT)
  - Evaluation metrics, analysis, and visualizations

## Citation

If you use this code or models in your research, please cite:

```bibtex
@inproceedings{madan2025predicting,
  title     = {Predicting Helpful Community Notes with Prompting and Parameter-Efficient Fine-Tuning},
  author    = {Salim, Md Shahidul and Madan, Mridul and Pashkov, Rafael},
  booktitle = {Social Computing},
  year      = {2025},
  institution = {University of Massachusetts Lowell},
  url       = {https://github.com/shahidul034/Predicting-Helpful-Community-Notes-with-Prompting-and-Parameter-Efficient-Fine-Tuning}
}
```

## References

1. Allen et al. "Birds of a Feather Don't Fact-Check Each Other." CHI 2022.
2. Chuai et al. "Did the Roll-Out of Community Notes Reduce Engagement?" CSCW 2024.
3. Saeed et al. "Crowdsourced Fact-Checking at Twitter." CIKM 2022.
4. Unsloth: Efficient LLM Fine-Tuning. https://github.com/unslothai/unsloth
