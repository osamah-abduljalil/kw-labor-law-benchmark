"""Abstract base interface for LLM backends."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract interface for language model generation."""

    def __init__(self, model_name: str, max_new_tokens: int = 500):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 500
    ) -> str:
        """Generates text from a prompt string.

        Args:
            prompt: Text prompt to feed the model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response string.
        """
        pass
