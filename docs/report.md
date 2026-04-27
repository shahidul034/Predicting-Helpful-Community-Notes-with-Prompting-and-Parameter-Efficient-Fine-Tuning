# Model Evaluation Report

## Overview

This report summarizes the saved evaluation artifacts in `outputs/evaluations` for prompting, supervised baselines, and LoRA fine-tuning experiments on Community Notes helpfulness prediction.

There are two evaluation settings in the saved outputs:

- Prompting results are based on saved JSON outputs with `100` predictions and `100` labels.
- The BERT baseline and newer Unsloth LoRA fine-tuning runs were evaluated on a separate `246`-sample test split.

Because the split sizes differ, prompting results should be compared mainly within the prompting section, while BERT and LoRA results should be compared within the `246`-sample setting.

## Prompting Results

### Zero-Shot


| Model                         | Accuracy | Precision | Recall | F1    |
| ----------------------------- | -------- | --------- | ------ | ----- |
| `google/gemma-3-4b-it`        | 45.53%   | 0.00%     | 0.00%  | 0.00% |
| `Qwen/Qwen3-4B-Instruct-2507` | 45.53%   | 0.00%     | 0.00%  | 0.00% |


Both zero-shot runs collapsed to predicting every sample as `NOT_HELPFUL`.

### Few-Shot

The corrected few-shot outputs are the `*_results_v2.json` files.


| Model                         | Accuracy | Precision | Recall | F1     |
| ----------------------------- | -------- | --------- | ------ | ------ |
| `google/gemma-3-4b-it`        | 52.44%   | 56.69%    | 53.73% | 55.17% |
| `Qwen/Qwen3-4B-Instruct-2507` | 49.19%   | 53.19%    | 55.97% | 54.55% |


After the decoding fix, few-shot prompting recovered to roughly `0.55` F1 for both models. Gemma is slightly stronger on accuracy, precision, and F1, while Qwen is slightly stronger on recall.

### Chain-of-Thought


| Model                         | Accuracy | Precision | Recall | F1     |
| ----------------------------- | -------- | --------- | ------ | ------ |
| `google/gemma-3-4b-it`        | 56.10%   | 58.40%    | 67.20% | 62.50% |
| `Qwen/Qwen3-4B-Instruct-2507` | 53.66%   | 56.17%    | 67.91% | 61.49% |


Chain-of-thought is the strongest prompting strategy in the saved outputs. Both models reach recall near `67%`, and Gemma has a small overall edge on accuracy, precision, and F1.

## Supervised Baseline

The BERT baseline in `outputs/evaluations/bert_baseline_metrics.json` was evaluated on `754` training samples and a `246`-sample test split.


| Model               | Test Samples | Accuracy | Precision | Recall | F1     | Eval Loss |
| ------------------- | ------------ | -------- | --------- | ------ | ------ | --------- |
| `bert-base-uncased` | 246          | 59.76%   | 62.96%    | 63.43% | 63.20% | 0.6581    |


Training configuration: 3 epochs, batch size `32`, learning rate `2e-5`, max length `512`, and weight decay `0.01`. The best checkpoint was selected by F1 at epoch 2 (`0.6296`).

## Unsloth LoRA Fine-Tuning

The newer saved LoRA evaluation metadata comes from:

- `outputs/evaluations/finetuned/gemma-3-4b-it/evaluation_report.json`
- `outputs/evaluations/finetuned/evaluation_report.json`

Both were evaluated on the same `246`-sample test split as the BERT baseline. An older Gemma summary is also still available in `outputs/evaluations/gemma-3-4b_unsloth_results.json`.


| Model                         | Method                    | Test Samples | Accuracy | Precision | Recall  | F1     | Train Loss | Train Runtime |
| ----------------------------- | ------------------------- | ------------ | -------- | --------- | ------- | ------ | ---------- | ------------- |
| `unsloth/gemma-3-4b-it`       | `LoRA_finetuning_unsloth` | 246          | 95.53%   | 93.01%    | 99.25%  | 96.03% | 0.3894     | 433.97s       |
| `Qwen/Qwen3-4B-Instruct-2507` | `LoRA_finetuning_unsloth` | 246          | 82.11%   | 75.28%    | 100.00% | 85.90% | 0.3871     | 329.11s       |








