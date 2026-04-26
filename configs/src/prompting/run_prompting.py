"""
Prompting Experiments for Community Notes Prediction.

Implements zero-shot, few-shot, and chain-of-thought prompting approaches.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_all_metrics

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompting experiments")
    parser.add_argument(
        "--method",
        type=str,
        choices=["zero-shot", "few-shot", "cot"],
        default="zero-shot",
        help="Prompting method",
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
        default="outputs/evaluations",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="Model name for prompting",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference"
    )
    return parser.parse_args()


# Prompt templates
ZERO_SHOT_PROMPT = """You are a fact-checking assistant. Given a tweet and a community note that provides context or correction, predict whether the note will be rated as "Helpful" by diverse users.

Tweet: {tweet}
Community Note: {note}

Will this note be rated as Helpful? Answer with only "HELPFUL" or "NOT_HELPFUL".

Prediction: """


FEW_SHOT_PROMPT = """You are a fact-checking assistant. Given a tweet and a community note, predict whether the note will be rated as "Helpful".

Example 1:
Tweet: "Vaccines cause autism in children!"
Community Note: "Multiple large-scale scientific studies have found no link between vaccines and autism. The original study claiming this link has been retracted due to serious methodological flaws."
Prediction: HELPFUL

Example 2:
Tweet: "The president just announced new climate policies."
Community Note: "This is obvious garbage and everyone who disagrees is stupid."
Prediction: NOT_HELPFUL

Example 3:
Tweet: "Climate change is a hoax created by the government."
Community Note: "According to NASA and the IPCC, there is overwhelming scientific consensus that climate change is real and primarily caused by human activities. Over 97% of climate scientists agree on this."
Prediction: HELPFUL

Example 4:
Tweet: "New study shows coffee reduces cancer risk."
Community Note: "I hate coffee so this is probably fake news."
Prediction: NOT_HELPFUL

Example 5:
Tweet: "The moon landing was faked in a studio."
Community Note: "The Apollo moon landings are supported by extensive evidence including lunar samples, independent tracking by multiple countries, and photographs from lunar orbiters. No credible evidence of a hoax exists."
Prediction: HELPFUL

Now, predict for the following:
Tweet: {tweet}
Community Note: {note}

Prediction: """


CHAIN_OF_THOUGHT_PROMPT = """You are a fact-checking assistant. Analyze the community note and explain your reasoning before predicting whether it will be rated as "Helpful".

Tweet: {tweet}
Community Note: {note}

Consider these factors:
1. Does the note cite credible sources?
2. Is the tone neutral and non-partisan?
3. Does it provide specific, factual information?
4. Is it clear and easy to understand?
5. Does it avoid personal attacks or inflammatory language?

Reasoning:
"""


def load_few_shot_examples() -> List[Dict]:
    """Load few-shot examples for in-context learning."""
    return [
        {
            "tweet": "Vaccines cause autism in children!",
            "note": "Multiple large-scale scientific studies have found no link between vaccines and autism.",
            "label": "HELPFUL",
        },
        {
            "tweet": "The president just announced new climate policies.",
            "note": "This is obvious garbage and everyone who disagrees is stupid.",
            "label": "NOT_HELPFUL",
        },
        {
            "tweet": "Climate change is a hoax created by the government.",
            "note": "According to NASA and the IPCC, there is overwhelming scientific consensus that climate change is real.",
            "label": "HELPFUL",
        },
        {
            "tweet": "New study shows coffee reduces cancer risk.",
            "note": "I hate coffee so this is probably fake news.",
            "label": "NOT_HELPFUL",
        },
        {
            "tweet": "The moon landing was faked in a studio.",
            "note": "The Apollo moon landings are supported by extensive evidence including lunar samples and independent tracking.",
            "label": "HELPFUL",
        },
    ]


def build_prompt(
    method: str, tweet: str, note: str, examples: List[Dict] = None
) -> str:
    """Build prompt based on method."""
    if method == "zero-shot":
        return ZERO_SHOT_PROMPT.format(tweet=tweet, note=note)
    elif method == "few-shot":
        return FEW_SHOT_PROMPT.format(tweet=tweet, note=note)
    elif method == "cot":
        return CHAIN_OF_THOUGHT_PROMPT.format(tweet=tweet, note=note)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_prediction(completion: str) -> str:
    """Extract HELPFUL or NOT_HELPFUL from model completion."""
    completion = completion.upper().strip()

    if "HELPFUL" in completion and "NOT" not in completion.split("HELPFUL")[0]:
        return "HELPFUL"
    elif "NOT_HELPFUL" in completion or "NOT HELPFUL" in completion:
        return "NOT_HELPFUL"
    elif "HELPFUL" in completion:
        return "HELPFUL"

    return "UNKNOWN"


def load_data(input_dir: Path, max_samples: int = None) -> pd.DataFrame:
    """Load test data."""
    test_df = pd.read_parquet(input_dir / "community_notes_cleaned_test.parquet")

    if max_samples:
        test_df = test_df.head(max_samples)

    # Use note text as tweet if no tweet column exists
    if "tweet_text" not in test_df.columns:
        test_df["tweet_text"] = test_df["summary"].str[:100] + "..."

    return test_df


def run_prompting(
    model, tokenizer, method: str, test_df: pd.DataFrame, batch_size: int
) -> List[str]:
    """Run prompting on test set."""
    predictions = []

    for i in tqdm(range(0, len(test_df), batch_size), desc="Running prompting"):
        batch = test_df.iloc[i : i + batch_size]

        for _, row in batch.iterrows():
            prompt = build_prompt(method, row["tweet_text"], row["summary"])

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            completion = tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            )
            prediction = extract_prediction(completion)
            predictions.append(prediction)

    return predictions


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting {args.method} prompting experiment...")

    # Load data
    test_df = load_data(input_dir, args.max_samples)
    y_true = test_df["label"].tolist()

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    logger.info(
        f"GPU: CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype="auto"
    )

    # Run prompting
    predictions = run_prompting(model, tokenizer, args.method, test_df, args.batch_size)

    # Convert predictions to numeric
    y_pred = [1 if p == "HELPFUL" else 0 for p in predictions]
    y_prob = [0.9 if p == "HELPFUL" else 0.1 for p in predictions]

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    print(f"\n=== {args.method.upper()} Prompting Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save results
    results = {
        "method": args.method,
        "model": args.model_name,
        "metrics": metrics,
        "predictions": predictions[:100],
        "true_labels": y_true[:100],
    }

    with open(output_dir / f"{args.method}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_dir / args.method}_results.json")
    logger.info(f"{args.method} prompting complete!")


if __name__ == "__main__":
    main()
