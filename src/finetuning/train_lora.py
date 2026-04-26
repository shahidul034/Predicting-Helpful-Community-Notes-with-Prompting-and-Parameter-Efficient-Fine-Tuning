"""
LoRA Fine-Tuning Script for Community Notes Prediction.

Fine-tunes Qwen3-4B or Gemma-3-4B with LoRA adapters using Unsloth.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_all_metrics

logger = setup_logging()
config = Config()

# Model mapping
MODEL_MAP = {
    "qwen3-4b": {
        "name": "unsloth/Qwen3-4B-Instruct-2507",
        "short": "qwen3-4b-instruct-2507",
    },
    "gemma-3-4b": {
        "name": "unsloth/gemma-3-4b-it",
        "short": "gemma-3-4b-it",
    },
}

PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Community Notes")
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen3-4b", "gemma-3-4b"],
        default="qwen3-4b",
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to train parquet file (default: auto from project root)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test parquet file (default: auto from project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default=None,
        help="Directory to save LoRA adapters",
    )
    return parser.parse_args()


def format_chat_example(row):
    """Format a single row as a chat example."""
    label_text = (
        "helpful and accurate" if row["label"] == 1 else "not helpful or inaccurate"
    )
    classification = row.get("classification", "unknown")
    messages = [
        {
            "role": "user",
            "content": (
                "You are a fact-checking assistant. Analyze the following community note "
                "summary and classify whether it is helpful and accurate or not helpful and inaccurate.\n\n"
                "Community note summary:\n"
                f"{row['summary']}\n\n"
                f"Classification: {classification}\n\n"
                "Is this community note helpful and accurate? Answer with only 'helpful and accurate' or 'not helpful and inaccurate'."
            ),
        },
        {
            "role": "assistant",
            "content": label_text,
        },
    ]
    return {"messages": messages}


def apply_chat_template(examples, tokenizer):
    """Apply chat template to batched examples."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


def load_datasets(train_path: str, test_path: str):
    """Load and format train/test datasets."""
    logger.info(f"Loading train data from {train_path}")
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    train_dataset = Dataset.from_list(
        train_df.apply(format_chat_example, axis=1).to_list()
    )
    test_dataset = Dataset.from_list(
        test_df.apply(format_chat_example, axis=1).to_list()
    )

    return train_dataset, test_dataset, train_df, test_df


def train(args):
    """Main training loop."""
    model_info = MODEL_MAP[args.model]
    model_name = model_info["name"]
    model_short = model_info["short"]

    # Resolve paths
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_data = args.train_data or str(data_dir / "community_notes_cleaned_train.parquet")
    test_data = args.test_data or str(data_dir / "community_notes_cleaned_test.parquet")

    output_dir = args.output_dir or str(
        PROJECT_ROOT / "outputs" / "evaluations" / "finetuned" / f"{model_short}"
    )
    model_save_dir = args.model_save_dir or str(
        PROJECT_ROOT / "outputs" / "models" / f"{model_short}-lora"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    logger.info(f"Model: {model_name}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Model save dir: {model_save_dir}")

    # Load datasets
    train_dataset, test_dataset, train_df, test_df = load_datasets(train_data, test_data)

    # Load model
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    logger.info("Adding LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Format datasets
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: apply_chat_template(x, tokenizer), batched=True
    )

    logger.info("Sample formatted prompt (truncated):")
    logger.info(train_dataset[0]["text"][:500])

    # Train
    logger.info("Starting training")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    effective_batch = args.batch_size // (args.gradient_accumulation_steps if args.batch_size >= args.gradient_accumulation_steps else 1)
    if effective_batch < 1:
        effective_batch = args.batch_size

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=effective_batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=output_dir,
            report_to="none",
            save_strategy="epoch",
            eval_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    start_time = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - start_time

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    logger.info(f"{round(elapsed, 1)} seconds used for training.")
    logger.info(f"{round(elapsed / 60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")

    # Save training metrics
    train_metrics = {k: float(v) for k, v in trainer_stats.metrics.items()}
    train_metrics["elapsed_seconds"] = elapsed
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Save LoRA adapters
    logger.info(f"Saving LoRA adapters to {model_save_dir}")
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logger.info(f"LoRA adapters saved to {model_save_dir}")

    # Evaluate
    logger.info("Evaluating on test set")
    FastLanguageModel.for_inference(model)

    BATCH_SIZE = 16
    correct = 0
    total = len(test_dataset)
    predictions_log = []

    test_prompts = []
    test_expected = []
    for sample in test_df.itertuples():
        label_text = (
            "helpful and accurate" if sample.label == 1 else "not helpful or inaccurate"
        )
        classification = sample.classification
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a fact-checking assistant. Analyze the following community note "
                    "summary and classify whether it is helpful and accurate or not helpful and inaccurate.\n\n"
                    "Community note summary:\n"
                    f"{sample.summary}\n\n"
                    f"Classification: {classification}\n\n"
                    "Is this community note helpful and accurate? Answer with only 'helpful and accurate' or 'not helpful and inaccurate'."
                ),
            }
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        test_prompts.append(prompt_text)
        test_expected.append(label_text)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        prompts_batch = test_prompts[batch_start:batch_end]
        expected_batch = test_expected[batch_start:batch_end]

        inputs = tokenizer(
            prompts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            use_cache=True,
            temperature=0.1,
        )

        for i, out_ids in enumerate(outputs):
            prompt_len = (
                inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
            )
            generated_ids = out_ids[prompt_len:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            expected = expected_batch[i]

            is_correct = False
            if expected == "helpful and accurate":
                is_correct = (
                    "helpful" in generated.lower() and "accurate" in generated.lower()
                )
            else:
                is_correct = (
                    "not helpful" in generated.lower()
                    or "inaccurate" in generated.lower()
                )

            if is_correct:
                correct += 1

            predictions_log.append(
                {
                    "index": batch_start + i,
                    "generated": generated,
                    "expected": expected,
                    "correct": is_correct,
                }
            )

    accuracy = correct / total * 100
    logger.info(f"Test Accuracy: {correct}/{total} = {accuracy:.2f}%")

    # Compute detailed metrics
    y_true = [1 if e == "helpful and accurate" else 0 for e in test_expected]
    y_pred = [1 if p["correct"] and p["expected"] == p["generated"] or
              (p["correct"] and "helpful" in p["generated"].lower()) else 0
              for p in predictions_log]

    tp = sum(
        1
        for p in predictions_log
        if p["correct"] and p["expected"] == "helpful and accurate"
    )
    fp = sum(
        1
        for p in predictions_log
        if not p["correct"] and p["expected"] == "not helpful or inaccurate"
    )
    fn = sum(
        1
        for p in predictions_log
        if not p["correct"] and p["expected"] == "helpful and accurate"
    )
    tn = sum(
        1
        for p in predictions_log
        if p["correct"] and p["expected"] == "not helpful or inaccurate"
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    logger.info(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save predictions
    preds_path = os.path.join(output_dir, "test_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predictions": predictions_log,
            },
            f,
            indent=2,
        )

    # Save evaluation report
    eval_report = {
        "model": model_name,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_samples": len(train_dataset),
        "test_samples": total,
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "training_metrics": train_metrics,
        "elapsed_seconds": elapsed,
    }
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info("=== DONE ===")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
