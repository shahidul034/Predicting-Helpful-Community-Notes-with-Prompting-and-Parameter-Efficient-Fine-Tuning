"""
vLLM Serving Script for Fine-Tuned Community Notes Models.

Starts a vLLM server for running inference on fine-tuned LoRA models.
"""

import os
import sys
import argparse
import subprocess
import signal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()
config = Config()

PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (base + LoRA adapters)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model length",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of tensor parallel replicas",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="Swap space size in GB",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode execution",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name/path (required if model-path is LoRA adapters only)",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path: relative to project root if not absolute."""
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def start_server(args):
    """Start vLLM server process."""
    model_path = resolve_model_path(args.model_path)

    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    logger.info(f"Starting vLLM server with model: {model_path}")
    logger.info(f"Host: {args.host}, Port: {args.port}")

    # Build vLLM command
    cmd = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--model", str(model_path),
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--swap-space", str(args.swap_space),
    ]

    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.base_model:
        cmd.extend(["--vision-model", args.base_model])

    logger.info(f"Command: {' '.join(cmd)}")

    def signal_handler(sig, frame):
        logger.info("Shutting down vLLM server...")
        if process.poll() is None:
            process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        process = subprocess.Popen(cmd)
        logger.info(f"vLLM server started (PID: {process.pid})")
        logger.info(f"Server available at http://{args.host}:{args.port}")
        process.wait()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    logger.info(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    start_server(args)


if __name__ == "__main__":
    main()
