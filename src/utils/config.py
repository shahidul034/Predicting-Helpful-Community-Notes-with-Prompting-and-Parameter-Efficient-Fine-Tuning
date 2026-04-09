import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        # Try relative path first, then absolute path from project root
        if not self.config_dir.exists():
            # Try from project root (parent of src/)
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / "configs"

        if not self.config_dir.exists():
            # Create empty configs
            self._configs = {
                "data": self._default_data_config(),
                "model": self._default_model_config(),
                "training": self._default_training_config(),
            }
            return

        for yaml_file in self.config_dir.glob("*.yaml"):
            config_name = yaml_file.stem
            with open(yaml_file, "r", encoding="utf-8") as f:
                self._configs[config_name] = yaml.safe_load(f)

    def _default_data_config(self) -> dict:
        return {
            "filtering": {"exclude_statuses": ["NEEDS_MORE_RATINGS"], "min_ratings": 5},
            "columns": {"status": "currentLabelStatus", "created_at": "createdAt"},
            "temporal_split": {
                "train_cutoff_date": "2024-10-01",
                "val_cutoff_date": "2024-11-01",
            },
            "label_mapping": {
                "CURRENTLY_RATED_HELPFUL": 1,
                "CURRENTLY_RATED_NOT_HELPFUL": 0,
            },
        }

    def _default_model_config(self) -> dict:
        return {}

    def _default_training_config(self) -> dict:
        return {}

    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        if config_name not in self._configs:
            return default
        if key is None:
            return self._configs[config_name]
        value = self._configs[config_name]
        for k in key.split("."):
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def __getitem__(self, config_name: str) -> Any:
        return self._configs.get(config_name, {})


config = Config()
