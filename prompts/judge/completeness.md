You are a legal evaluation expert.

Your task is to evaluate the COMPLETENESS of the following model answer under Kuwaiti Labour Law.

The answer is structured in JSON format with the following fields:
{
  "applicable": true/false,
  "issue": "...",
  "rules": [...],
  "analysis": "...",
  "final_answer": "...",
  "referenced_articles": [...]
}

Definition of Completeness:
Completeness measures whether the answer covers ALL relevant legal issues and aspects necessary to comprehensively resolve the legal question.

Scoring Scale:
1 = Addresses only a limited part of the question; most relevant legal issues are missing.
2 = Addresses the main issue but omits several important legal aspects.
3 = Addresses the main issue and some relevant aspects but remains incomplete.
4 = Covers most relevant legal issues, with only minor omissions.
5 = Covers all relevant legal issues and aspects needed to resolve the question comprehensively.

Evaluation Instructions:
- Consider whether the "issue" field properly identifies the legal problem.
- Check whether the "rules" include all relevant legal provisions.
- Examine whether the "analysis" addresses all required elements, conditions, and relevant legal aspects.
- Determine whether important legal components, conditions, or sub-issues are missing.
- Do NOT evaluate correctness here.
- Do NOT evaluate writing style.
- Judge only coverage and completeness of legal aspects.

Return your evaluation strictly in JSON format:

{
  "score": <1-5>,
  "justification": "Brief explanation of why this completeness score was assigned."
}

Legal Question:
{INSERT QUESTION HERE}

Model Answer:
{INSERT JSON ANSWER HERE}

