"""API-based LLM providers (DeepSeek, Google Gemini, OpenAI)."""

import os
from typing import Optional
from .base import BaseLLM

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    retry = None


class DeepSeekLLM(BaseLLM):
    """DeepSeek API model backend via OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        max_new_tokens: int = 500
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        if OpenAI is None:
            raise ImportError("openai package must be installed to use DeepSeekLLM.")

        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key or key == "your_deepseek_api_key_here":
            raise ValueError("DEEPSEEK_API_KEY not found in environment or .env file.")

        self.client = OpenAI(api_key=key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 500
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|endoftext|>"]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response from DeepSeek API: {e}"


class GeminiLLM(BaseLLM):
    """Google Gemini API model backend with exponential retry."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_new_tokens: int = 500
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        if genai is None:
            raise ImportError("google-generativeai must be installed to use GeminiLLM.")

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key or key == "your_gemini_api_key_here":
            raise ValueError("GOOGLE_API_KEY not found in environment or .env file.")

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 500
    ) -> str:
        def _call_api():
            config = genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            response = self.model.generate_content(prompt, generation_config=config)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            raise RuntimeError("No content returned by Gemini API.")

        if retry is not None:
            _call_with_retry = retry(
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(5),
                retry=retry_if_exception_type(Exception)
            )(_call_api)
        else:
            _call_with_retry = _call_api

        try:
            return _call_with_retry()
        except Exception as e:
            return f"Error generating response from Gemini API: {e}"


class OpenAILLM(BaseLLM):
    """Standard OpenAI and OpenAI-compatible API model backend (GPT-4o, OpenRouter, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_new_tokens: int = 500
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        if OpenAI is None:
            raise ImportError("openai package must be installed to use OpenAILLM.")

        key = api_key or os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
        if not key or key == "your_openai_api_key_here":
            raise ValueError(f"{api_key_env} not found in environment or .env file.")

        client_kwargs = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 500
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response from OpenAI API: {e}"
