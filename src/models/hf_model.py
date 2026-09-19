"""Hugging Face local causal language model wrapper."""

import os
from typing import Optional
from .base import BaseLLM

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class HuggingFaceLLM(BaseLLM):
    """Local HuggingFace causal LM backend (e.g., ALLaM, LLaMA)."""

    def __init__(
        self,
        model_id: str = "humain-ai/ALLaM-7B-Instruct-preview",
        load_4bit: bool = True,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 500,
        device_map: str = "auto"
    ):
        super().__init__(model_name=model_id, max_new_tokens=max_new_tokens)

        if torch is None or AutoModelForCausalLM is None:
            raise ImportError("torch and transformers must be installed to use HuggingFaceLLM.")

        token = hf_token or os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            use_fast=True
        )

        load_kwargs = {
            "token": token,
            "torch_dtype": torch.float16,
        }

        # Check if CUDA and bitsandbytes are available for 4-bit
        if torch.cuda.is_available():
            load_kwargs["device_map"] = device_map
            if load_4bit:
                try:
                    import bitsandbytes
                    load_kwargs["load_in_4bit"] = True
                except ImportError:
                    pass
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 500
    ) -> str:
        """Generates response using the local HuggingFace model."""
        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the input prompt from the output
        if full_response.startswith(prompt):
            return full_response[len(prompt):].strip()
        return full_response.strip()
