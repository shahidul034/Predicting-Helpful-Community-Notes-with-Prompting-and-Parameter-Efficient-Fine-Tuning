"""
LoRA Fine-Tuning with Unsloth.

Efficient fine-tuning of LLMs for Community Notes effectiveness prediction.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM with LoRA using Unsloth"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen3-4b", "gemma-3-4b"],
        default="qwen3-4b",
        help="Model to fine-tune",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    return parser.parse_args()


def get_model_name(model_choice: str) -> str:
    """Get model name from choice."""
    model_names = {
        "qwen3-4b": "Qwen/Qwen3-4B-Instruct",
        "gemma-3-4b": "google/gemma-3-4b-it",
    }
    return model_names.get(model_choice, model_names["qwen3-4b"])


def load_data(input_dir: Path) -> tuple:
    """Load train and validation data."""
    train_df = pd.read_parquet(input_dir / "community_notes_cleaned_train.parquet")
    val_df = pd.read_parquet(input_dir / "community_notes_cleaned_val.parquet")

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    return train_df, val_df


def format_training_example(row) -> str:
    """Format a single training example as a chat conversation."""
    label_text = "HELPFUL" if row["label"] == 1 else "NOT_HELPFUL"

    # Use note text as tweet context if no tweet column exists
    tweet = row.get("tweet_text", row["summary"][:100] + "...")

    conversation = f"""<|im_start|>system
You are a fact-checking assistant. Analyze community notes and predict whether they will be rated as helpful.</s>
<|im_start|>user
I will provide a tweet and a community note. Predict whether the note will be rated as HELPFUL or NOT_HELPFUL.

Tweet: {tweet}
Community Note: {row["summary"]}

Will this note be rated as Helpful? Answer with only "HELPFUL" or "NOT_HELPFUL".</s>
<|im_start|>assistant
{label_text}</s>"""

    return conversation


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Prepare dataset for training."""
    dataset = Dataset.from_pandas(df)
    formatted = dataset.map(lambda x: {"text": format_training_example(x)})
    return formatted


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name(args.model)
    logger.info(f"Starting LoRA fine-tuning with {model_name}...")

    # Load model with Unsloth
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Configure LoRA
    logger.info("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load and prepare data
    train_df, val_df = load_data(input_dir)
    train_dataset = prepare_dataset(train_df)
    val_dataset = prepare_dataset(val_df)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / f"{args.model}_lora"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        fp16=True,
        bf16=False,
        report_to="none",
        device="cuda:0",  # Maps to GPU 2 due to CUDA_VISIBLE_DEVICES=2
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        args=training_args,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    save_path = output_dir / f"{args.model}_lora"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info(f"Saved fine-tuned model to {save_path}")
    logger.info(f"{args.model} LoRA fine-tuning complete!")


if __name__ == "__main__":
    main()
