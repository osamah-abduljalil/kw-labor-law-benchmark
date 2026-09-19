"""LLM-as-a-judge automated evaluator for legal benchmark."""

import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union
from models.base import BaseLLM


class LegalBenchmarkJudge:
    """Evaluates generated legal answers against benchmark rubrics (Completeness, Correctness, Fluency)."""

    METRICS = ["completeness", "correctness", "fluency"]

    def __init__(
        self,
        evaluator_llm: BaseLLM,
        prompts_dir: Optional[Union[str, Path]] = None
    ):
        self.llm = evaluator_llm
        if prompts_dir is None:
            # Check candidate paths
            candidates = [
                Path("prompts/judge"),
                Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "judge",
            ]
            for c in candidates:
                if c.exists():
                    prompts_dir = c
                    break

        self.prompts_dir = Path(prompts_dir) if prompts_dir else Path("prompts/judge")
        self.rubrics: Dict[str, str] = {}
        self._load_rubrics()

    def _load_rubrics(self):
        """Loads prompt templates for completeness, correctness, and fluency."""
        for metric in self.METRICS:
            fpath = self.prompts_dir / f"{metric}.md"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    self.rubrics[metric] = f.read()

    def evaluate_metric(
        self,
        question: str,
        answer: str,
        metric: str
    ) -> Dict[str, Any]:
        """Evaluates a single metric for a question/answer pair.

        Returns:
            Dict containing 'score' (int or float), 'justification' (str), and 'raw_response'.
        """
        metric_key = metric.lower()
        if metric_key not in self.rubrics:
            raise ValueError(f"Rubric for metric '{metric}' not found in {self.prompts_dir}")

        template = self.rubrics[metric_key]
        prompt = template.replace("{INSERT QUESTION HERE}", question)
        prompt = prompt.replace("{INSERT JSON ANSWER HERE}", answer)
        prompt = prompt.replace("{INSERT JSsON ANSWER HERE}", answer)

        raw_output = self.llm.generate(prompt, temperature=0.0, max_new_tokens=400)

        # Parse JSON output from the evaluator
        parsed = self._parse_judge_json(raw_output)
        parsed["raw_response"] = raw_output
        parsed["metric"] = metric_key
        return parsed

    @staticmethod
    def _parse_judge_json(text: str) -> Dict[str, Any]:
        """Extracts score and justification from LLM response."""
        # Try direct JSON parsing
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # Try regex search for JSON block
        json_match = re.search(r"\{[\s\S]*?\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass

        # Fallback: regex search for score and justification fields
        score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        just_match = re.search(r'"justification"\s*:\s*"([^"]+)"', text)

        score = float(score_match.group(1)) if score_match else None
        justification = just_match.group(1) if just_match else text.strip()

        return {
            "score": score,
            "justification": justification
        }
