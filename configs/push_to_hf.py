#!/usr/bin/env python3
"""Push local LoRA adapter folders directly to Hugging Face."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token

    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()

    print("ERROR: No HF_TOKEN found. Run 'huggingface-cli login' first.")
    sys.exit(1)


def validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    required_files = ["adapter_config.json", "README.md"]
    missing_files = [name for name in required_files if not (adapter_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {adapter_dir}: {', '.join(missing_files)}"
        )

    adapter_weight_patterns = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "*.safetensors",
        "*.bin",
    )
    has_adapter_weights = any(
        any(adapter_dir.glob(pattern)) for pattern in adapter_weight_patterns
    )
    if not has_adapter_weights:
        raise FileNotFoundError(
            "No adapter weights found in "
            f"{adapter_dir}. Expected something like adapter_model.safetensors."
        )


HF_TOKEN = get_hf_token()

try:
    info = whoami(token=HF_TOKEN)
    print(f"Logged in as: {info['name']}")
except Exception as exc:
    print(f"ERROR: Cannot authenticate with HF: {exc}")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

models = [
    {
        "adapter_dir": "/home/mshahidul/llm_safety/sc_project/outputs/models/gemma-3-4b_unsloth_lora",
        "base_model": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "repo_name": "gemma-3-4b-it-CommunityNotesEffectiveness",
    },
    {
        "adapter_dir": "/home/mshahidul/llm_safety/sc_project/outputs/models/qwen3-4b-instruct-2507_unsloth_lora",
        "base_model": "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit",
        "repo_name": "qwen3-4b-instruct-2507-CommunityNotesEffectiveness",
    },
]

for idx, cfg in enumerate(models, start=1):
    print(f"\n{'=' * 60}")
    print(f"Model {idx}/{len(models)}: {cfg['repo_name']}")
    print(f"{'=' * 60}")

    adapter_dir = Path(cfg["adapter_dir"])
    base_model = cfg["base_model"]
    repo_name = cfg["repo_name"]
    repo_id = f"shahidul034/{repo_name}"

    print(f"Preparing direct upload from: {adapter_dir}")
    print(f"Base model referenced by adapter: {base_model}")

    try:
        validate_adapter_dir(adapter_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Pushing adapter files to hf.co/{repo_id} ...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as exc:
        print(f"Note: repo creation returned: {exc}")

    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "checkpoint-*/*", "trainer_state.json"],
    )

    print(f"SUCCESS: https://huggingface.co/{repo_id}\n")

print("All adapter folders pushed successfully!")
