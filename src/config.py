"""Configuration and environment loader for Kuwaiti Labor Law Benchmark."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
    # Automatically look for .env in current directory or repo root
    load_dotenv()
except ImportError:
    load_dotenv = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "law_csv_path": "data/samples/sample_law_articles.csv",
        "qa_csv_path": "data/samples/sample_cases.csv",
        "output_dir": "data/generated",
        "index_dir": "data/index"
    },
    "data": {
        "law_text_col": "article_text",
        "law_id_col": "article_number",
        "q_col": "case_text",
        "chunking": {
            "enabled": False,
            "chunk_words": 220,
            "overlap_words": 40
        }
    },
    "retrieval": {
        "emb_name": "intfloat/multilingual-e5-large",
        "normalize_emb": True,
        "top_k_dense": 30,
        "top_k_final": 8,
        "use_bm25": True,
        "alpha_dense": 0.65
    },
    "generation": {
        "force_citations": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 500
    },
    "evaluation": {
        "evaluator_model": "deepseek-chat",
        "metrics": ["completeness", "correctness", "fluency"]
    },
    "models": {
        "gpt4o": {
            "type": "openai",
            "model_name": "gpt-4o",
            "max_new_tokens": 500,
            "enabled": True
        },
        "deepseek_api": {
            "type": "deepseek_api",
            "model_name": "deepseek-chat",
            "max_new_tokens": 500,
            "enabled": True
        },
        "llama31_openrouter": {
            "type": "openrouter",
            "model_name": "meta-llama/llama-3.1-70b-instruct",
            "max_new_tokens": 500,
            "enabled": True
        },
        "qwen": {
            "type": "openrouter",
            "model_name": "qwen/qwen-2.5-72b-instruct",
            "max_new_tokens": 500,
            "enabled": True
        },
        "gemini_api": {
            "type": "gemini_api",
            "model_name": "gemini-2.5-flash",
            "max_new_tokens": 500,
            "enabled": True
        },
        "allam_hf": {
            "type": "hf",
            "hf_id": "humain-ai/ALLaM-7B-Instruct-preview",
            "load_4bit": True,
            "max_new_tokens": 500,
            "enabled": False
        }
    }
}


class ConfigSection:
    """Helper class to access nested dictionaries via attribute access."""
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigSection(v))
            else:
                setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getattr__(self, name: str) -> Any:
        return None

    def to_dict(self) -> Dict[str, Any]:
        res = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigSection):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res

    def __repr__(self) -> str:
        return f"ConfigSection({self.__dict__})"


class BenchmarkConfig:
    """Benchmark configuration container."""

    def __init__(self, raw_config: Dict[str, Any]):
        # Merge default config with raw config
        merged = dict(DEFAULT_CONFIG)
        for k, v in raw_config.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                sub = dict(merged[k])
                sub.update(v)
                merged[k] = sub
            else:
                merged[k] = v

        self.paths = ConfigSection(merged.get("paths", {}))
        self.data = ConfigSection(merged.get("data", {}))
        self.retrieval = ConfigSection(merged.get("retrieval", {}))
        self.generation = ConfigSection(merged.get("generation", {}))
        self.evaluation = ConfigSection(merged.get("evaluation", {}))
        self.models = merged.get("models", {})
        self._raw = merged

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


def load_config(config_path: Optional[str] = None) -> BenchmarkConfig:
    """Load configuration from a YAML file. Defaults to config/default_config.yaml."""
    if config_path is None:
        # Check standard locations
        candidates = [
            Path("config/default_config.yaml"),
            Path(__file__).resolve().parent.parent.parent / "config" / "default_config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    raw = {}
    if config_path and Path(config_path).exists():
        if yaml is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

    return BenchmarkConfig(raw)
