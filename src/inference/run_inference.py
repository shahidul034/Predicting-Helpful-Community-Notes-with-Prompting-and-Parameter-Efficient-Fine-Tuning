"""
Run Inference on Community Notes Data.

Sends requests to a vLLM endpoint and evaluates model predictions.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import requests
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_all_metrics

logger = setup_logging()
config = Config()

PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference via vLLM endpoint")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="vLLM server endpoint URL",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to test parquet file (default: auto from project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for API requests"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Request timeout in seconds"
    )
    return parser.parse_args()


def build_prompt(row) -> str:
    """Build prompt for a single community note."""
    classification = row.get("classification", "unknown")
    return (
        "You are a fact-checking assistant. Analyze the following community note "
        "summary and classify whether it is helpful and accurate or not helpful and inaccurate.\n\n"
        "Community note summary:\n"
        f"{row['summary']}\n\n"
        f"Classification: {classification}\n\n"
        "Is this community note helpful and accurate? Answer with only 'helpful and accurate' or 'not helpful and inaccurate'.\n\n"
        "Prediction: "
    )


def query_vllm(
    endpoint: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    timeout: int,
) -> List[str]:
    """Query vLLM server for completions."""
    url = f"{endpoint}/generate"
    completions = []

    for prompt in prompts:
        payload = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 50,
        }

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            text = result.get("text", [{}])[0].get("text", "").strip()
            completions.append(text)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            completions.append("")
        except (IndexError, KeyError) as e:
            logger.error(f"Response parsing error: {e}")
            completions.append("")

    return completions


def extract_label(completion: str) -> int:
    """Extract binary label from model completion."""
    completion_lower = completion.lower().strip()

    if "helpful" in completion_lower and "accurate" in completion_lower:
        if "not" in completion_lower.split("helpful")[0]:
            return 0
        return 1
    elif "not helpful" in completion_lower or "inaccurate" in completion_lower:
        return 0

    if "helpful" in completion_lower:
        return 1
    return 0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    input_file = args.input_file or str(
        PROJECT_ROOT / "data" / "processed" / "community_notes_cleaned_test.parquet"
    )

    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info(f"Loading test data from {input_file}")
    test_df = pd.read_parquet(input_file)
    logger.info(f"Test samples: {len(test_df)}")

    # Check endpoint connectivity
    try:
        health_url = f"{args.endpoint}/health"
        resp = requests.get(health_url, timeout=5)
        logger.info(f"vLLM server is healthy: {resp.json()}")
    except requests.exceptions.RequestException:
        logger.warning("Could not reach vLLM server. Proceeding anyway...")

    # Build prompts
    prompts = [build_prompt(row) for _, row in test_df.iterrows()]
    y_true = test_df["label"].tolist()

    # Run inference in batches
    logger.info(f"Running inference in batches of {args.batch_size}")
    all_predictions = []
    all_completions = []
    total = len(prompts)

    start_time = time.time()

    for batch_start in tqdm(
        range(0, total, args.batch_size), desc="Running inference"
    ):
        batch_end = min(batch_start + args.batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]

        completions = query_vllm(
            args.endpoint,
            batch_prompts,
            args.max_new_tokens,
            args.temperature,
            args.timeout,
        )

        predictions = [extract_label(c) for c in completions]
        all_predictions.extend(predictions)
        all_completions.extend(completions)

    elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed:.1f} seconds")

    # Compute metrics
    metrics = compute_all_metrics(
        y_true, all_predictions, [0.9 if p == 1 else 0.1 for p in all_predictions]
    )

    print(f"\n=== Inference Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save predictions with details
    predictions_detail = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        predictions_detail.append(
            {
                "index": i,
                "summary": row["summary"],
                "true_label": int(row["label"]),
                "prediction": all_predictions[i] if i < len(all_predictions) else -1,
                "completion": all_completions[i] if i < len(all_completions) else "",
                "correct": all_predictions[i] == int(row["label"]) if i < len(all_predictions) else False,
            }
        )

    # Save results
    results = {
        "endpoint": args.endpoint,
        "metrics": metrics,
        "total_samples": total,
        "elapsed_seconds": elapsed,
        "predictions_detail": predictions_detail,
    }

    results_file = output_dir / "vllm_inference_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
