"""Factory to instantiate LLM models based on configuration."""

import logging
from typing import Any, Dict
from .api_models import DeepSeekLLM, GeminiLLM, OpenAILLM
from .base import BaseLLM
from .hf_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


def build_model(model_config: Dict[str, Any]) -> BaseLLM:
    """Builds a single LLM instance from a configuration dictionary."""
    m_type = model_config.get("type", "").lower()
    max_tokens = model_config.get("max_new_tokens", 500)

    if m_type == "deepseek_api":
        return DeepSeekLLM(
            model_name=model_config.get("model_name", "deepseek-chat"),
            max_new_tokens=max_tokens
        )
    elif m_type == "gemini_api":
        return GeminiLLM(
            model_name=model_config.get("model_name", "gemini-2.5-flash"),
            max_new_tokens=max_tokens
        )
    elif m_type in ("openai_api", "openai", "gpt"):
        return OpenAILLM(
            model_name=model_config.get("model_name", "gpt-4o"),
            api_key=model_config.get("api_key"),
            base_url=model_config.get("base_url"),
            api_key_env=model_config.get("api_key_env", "OPENAI_API_KEY"),
            max_new_tokens=max_tokens
        )
    elif m_type in ("openrouter_api", "openrouter"):
        return OpenAILLM(
            model_name=model_config.get("model_name", "meta-llama/llama-3.1-70b-instruct"),
            api_key=model_config.get("api_key"),
            base_url=model_config.get("base_url", "https://openrouter.ai/api/v1"),
            api_key_env=model_config.get("api_key_env", "OPENROUTER_API_KEY"),
            max_new_tokens=max_tokens
        )
    elif m_type in ("hf", "huggingface"):
        return HuggingFaceLLM(
            model_id=model_config.get("hf_id", "humain-ai/ALLaM-7B-Instruct-preview"),
            load_4bit=model_config.get("load_4bit", True),
            max_new_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported model type: '{m_type}'")


def load_models_from_config(models_dict: Dict[str, Any]) -> Dict[str, BaseLLM]:
    """Instantiates enabled models specified in the configuration."""
    loaded = {}
    for name, cfg in models_dict.items():
        # Check if enabled flag is false
        if isinstance(cfg, dict) and not cfg.get("enabled", True):
            continue

        try:
            model = build_model(cfg)
            loaded[name] = model
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.warning(f"Skipping model {name}: {e}")

    return loaded
