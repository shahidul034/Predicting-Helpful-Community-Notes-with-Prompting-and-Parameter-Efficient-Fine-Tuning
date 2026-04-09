"""
Run Inference on Test Data.

Evaluate fine-tuned models on the test set using vLLM or local inference.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_all_metrics, print_classification_report

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Inference mode: local or API",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to fine-tuned model (for local mode)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="API endpoint (for API mode)",
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
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    return parser.parse_args()


def build_prompt(tweet: str, note: str) -> str:
    """Build prompt for prediction."""
    return f"""<|im_start|>system
You are a fact-checking assistant. Analyze community notes and predict whether they will be rated as helpful.</s>
<|im_start|>user
I will provide a tweet and a community note. Predict whether the note will be rated as HELPFUL or NOT_HELPFUL.

Tweet: {tweet}
Community Note: {note}

Will this note be rated as Helpful? Answer with only "HELPFUL" or "NOT_HELPFUL".</s>
<|im_start|>assistant
"""


def extract_prediction(completion: str) -> str:
    """Extract HELPFUL or NOT_HELPFUL from completion."""
    completion = completion.upper().strip()

    if "HELPFUL" in completion and "NOT" not in completion.split("HELPFUL")[0]:
        return "HELPFUL"
    elif "NOT_HELPFUL" in completion or "NOT HELPFUL" in completion:
        return "NOT_HELPFUL"

    return "UNKNOWN"


def load_data(input_dir: Path) -> pd.DataFrame:
    """Load test data."""
    test_df = pd.read_parquet(input_dir / "community_notes_cleaned_test.parquet")

    if "tweet_text" not in test_df.columns:
        test_df["tweet_text"] = test_df["summary"].str[:100] + "..."

    return test_df


def run_local_inference(
    model_path: str, test_df: pd.DataFrame, batch_size: int
) -> List[str]:
    """Run inference locally using HuggingFace."""
    logger.info("Loading model for local inference...")
    logger.info(
        f"GPU: CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.1,
    )

    predictions = []

    for i in tqdm(range(0, len(test_df), batch_size), desc="Running inference"):
        batch = test_df.iloc[i : i + batch_size]

        for _, row in batch.iterrows():
            prompt = build_prompt(row["tweet_text"], row["summary"])

            result = pipe(prompt, pad_token_id=tokenizer.eos_token_id)
            completion = result[0]["generated_text"][len(prompt) :]

            prediction = extract_prediction(completion)
            predictions.append(prediction)

    return predictions


def run_api_inference(
    endpoint: str, test_df: pd.DataFrame, batch_size: int
) -> List[str]:
    """Run inference via API."""
    base_url = endpoint.rstrip("/")
    predictions = []

    for i in tqdm(range(0, len(test_df), batch_size), desc="Running API inference"):
        batch = test_df.iloc[i : i + batch_size]

        requests_data = {
            "items": [
                {"tweet": row["tweet_text"], "note": row["summary"]}
                for _, row in batch.iterrows()
            ]
        }

        response = requests.post(
            f"{base_url}/predict/batch", json=requests_data, timeout=120
        )

        if response.status_code == 200:
            results = response.json()
            predictions.extend([r["prediction"] for r in results["results"]])
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            predictions.extend(["UNKNOWN"] * len(batch))

    return predictions


def evaluate_predictions(y_true: List[int], predictions: List[str]) -> dict:
    """Evaluate predictions."""
    y_pred = [1 if p == "HELPFUL" else 0 for p in predictions]
    y_prob = [0.9 if p == "HELPFUL" else 0.1 for p in predictions]

    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    print("\n=== Inference Results ===")
    print(print_classification_report(y_true, y_pred))

    return metrics


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting inference...")

    # Load test data
    test_df = load_data(input_dir)
    y_true = test_df["label"].tolist()

    logger.info(f"Test set size: {len(test_df)}")

    # Run inference
    if args.mode == "local":
        if not args.model_path:
            raise ValueError("--model-path required for local mode")
        predictions = run_local_inference(args.model_path, test_df, args.batch_size)
    else:
        predictions = run_api_inference(args.endpoint, test_df, args.batch_size)

    # Evaluate
    metrics = evaluate_predictions(y_true, predictions)

    # Save results
    results = {
        "mode": args.mode,
        "model_path": args.model_path,
        "endpoint": args.endpoint,
        "metrics": metrics,
        "predictions": predictions[:100],
        "true_labels": y_true[:100],
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"inference_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_file}")
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
