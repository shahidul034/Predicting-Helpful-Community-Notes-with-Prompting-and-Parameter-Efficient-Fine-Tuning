"""
vLLM Server for serving fine-tuned models.

High-throughput inference server for Community Notes prediction.
"""

import os
import sys
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size"
    )
    return parser.parse_args()


class PredictionRequest(BaseModel):
    tweet: str
    note: str
    return_probabilities: bool = False


class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


app = FastAPI(title="Community Notes Prediction API")

# Global model instance
model_instance = None


def load_model(
    model_path: str, gpu_memory_utilization: float, tensor_parallel_size: int
):
    """Load vLLM model."""
    global model_instance

    if LLM is None:
        raise ImportError("vLLM is not installed. Install with: pip install vllm")

    logger.info(f"Loading model from {model_path}")
    model_instance = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=512,
        swap_space=4,
        enforce_eager=False,
    )
    logger.info("Model loaded successfully")


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


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    args = parse_args()
    load_model(args.model_path, args.gpu_memory_utilization, args.tensor_parallel_size)


@app.get("/")
async def root():
    return {"status": "running", "model": "Community Notes Predictor"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = build_prompt(request.tweet, request.note)

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50,
        stop_token_ids=[tokenizer.eos_token_id] if "tokenizer" in dir() else None,
    )

    outputs = model_instance.generate([prompt], sampling_params)
    completion = outputs[0].outputs[0].text.strip()

    # Extract prediction
    if "HELPFUL" in completion.upper() and "NOT" not in completion.split("HELPFUL")[0]:
        prediction = "HELPFUL"
    elif "NOT_HELPFUL" in completion.upper() or "NOT HELPFUL" in completion.upper():
        prediction = "NOT_HELPFUL"
    else:
        prediction = "UNKNOWN"

    return PredictionResponse(
        prediction=prediction, confidence=0.9 if prediction != "UNKNOWN" else 0.5
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    prompts = [build_prompt(item.tweet, item.note) for item in request.items]

    sampling_params = SamplingParams(temperature=0.1, max_tokens=50)

    outputs = model_instance.generate(prompts, sampling_params)

    for output, item in zip(outputs, request.items):
        completion = output.outputs[0].text.strip()

        if (
            "HELPFUL" in completion.upper()
            and "NOT" not in completion.split("HELPFUL")[0]
        ):
            prediction = "HELPFUL"
        elif "NOT_HELPFUL" in completion.upper() or "NOT HELPFUL" in completion.upper():
            prediction = "NOT_HELPFUL"
        else:
            prediction = "UNKNOWN"

        results.append(
            PredictionResponse(
                prediction=prediction,
                confidence=0.9 if prediction != "UNKNOWN" else 0.5,
            )
        )

    return BatchPredictionResponse(results=results)


def main():
    args = parse_args()

    logger.info(f"Starting vLLM server on {args.host}:{args.port}")
    logger.info(f"Model path: {args.model_path}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
