You are a legal evaluation expert.

Your task is to evaluate the CORRECTNESS of the following model answer under Kuwaiti Labour Law.

The answer is structured in JSON format with the following fields:
{
  "applicable": true/false,
  "issue": "...",
  "rules": [...],
  "analysis": "...",
  "final_answer": "...",
  "referenced_articles": [...]
}

Definition of Correctness:
Correctness measures whether the legal reasoning and conclusions are aligned with the applicable law and properly interpreted and applied.

Scoring Scale:
1 = Legal reasoning is incorrect or contradicts the applicable law.
2 = Legal reasoning is substantially incorrect, with major errors in interpretation or application.
3 = Legal reasoning is partially correct but includes noticeable legal errors.
4 = Legal reasoning is mostly correct, with only minor inaccuracies.
5 = Legal reasoning is fully correct and aligned with applicable law.

Evaluation Instructions:
- Evaluate whether the legal rules cited are accurate.
- Assess whether the interpretation of legal provisions is correct.
- Examine whether the "analysis" correctly applies the law to the facts.
- Determine whether the "final_answer" logically follows from the legal reasoning.
- Consider misinterpretation, misapplication, or contradiction of statutory provisions.
- Do NOT evaluate completeness.
- Do NOT evaluate writing style.

Return your evaluation strictly in JSON format:

{
  "score": <1-5>,
  "justification": "Brief explanation of why this correctness score was assigned."
}

Legal Question:
{INSERT QUESTION HERE}

Model Answer:
{INSERT JSsON ANSWER HERE}