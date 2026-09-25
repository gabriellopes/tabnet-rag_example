# app/src/utils/config_loader.py
import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def load_rag_config(cls) -> Dict[str, Any]:
        return cls._load_yaml("app/cfgs/rag.yaml").get("rag", {})

    @classmethod
    def load_reranker_config(cls) -> Dict[str, Any]:
        return cls._load_yaml("app/cfgs/plugins/reranker.yaml").get("reranker", {})

    @classmethod
    def load_finetuner_config(cls) -> Dict[str, Any]:
        return cls._load_yaml("app/cfgs/plugins/finetuner.yaml").get("finetuner", {})