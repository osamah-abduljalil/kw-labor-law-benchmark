"""Prompt construction templates for RAG and non-RAG legal generation."""

from typing import Any, Dict, List, Tuple


def _is_not_empty(val: Any) -> bool:
    if val is None:
        return False
    try:
        import pandas as pd
        if pd.isna(val):
            return False
    except ImportError:
        pass
    if isinstance(val, str) and not val.strip():
        return False
    return True


def format_evidence_block(text: str, meta: Dict[str, Any], k: int) -> str:
    """Formats a single retrieved evidence chunk with citation identifier [Ek]."""
    parts = [f"[E{k}]"]
    if "law_name" in meta and _is_not_empty(meta["law_name"]):
        parts.append(f"القانون: {meta['law_name']}")
    if "article" in meta and _is_not_empty(meta["article"]):
        parts.append(f"المادة: {meta['article']}")
    if "title" in meta and _is_not_empty(meta["title"]):
        parts.append(f"العنوان: {meta['title']}")

    header = " | ".join(parts)
    return f"{header}\n{text}"


def build_rag_prompt_ar(
    question: str,
    evidences: List[Tuple[str, Dict[str, Any]]],
    force_citations: bool = True
) -> str:
    """Constructs Arabic RAG prompt grounded strictly in retrieved evidence.

    Args:
        question: Legal question or scenario text.
        evidences: List of tuples (chunk_text, meta_dict).
        force_citations: Whether to mandate [E1], [E2] citation markers.

    Returns:
        Formatted prompt string.
    """
    evidence_blocks = [
        format_evidence_block(chunk_text, meta, i + 1)
        for i, (chunk_text, meta) in enumerate(evidences)
    ]
    evidence_text = "\n\n".join(evidence_blocks)

    citation_rule = ""
    if force_citations:
        citation_rule = (
            "التزم بالاستشهاد داخل الإجابة باستخدام معرف الدليل مثل [E1] أو [E2] بعد كل جملة قانونية.\n"
            "إذا لم تجد دليلاً كافياً في النصوص المقدمة، قل صراحة: (لا توجد مادة صريحة في الأدلة المقدمة).\n"
        )

    prompt = f"""أنت مستشار قانوني متخصص في القوانين الكويتية. أجب فقط اعتماداً على الأدلة المقدمة.

{citation_rule}
الأدلة:
{evidence_text}

السؤال:
{question}

الإجابة (بالعربية الفصحى، مختصرة ودقيقة):
"""
    return prompt


def build_non_rag_prompt_ar(question: str) -> str:
    """Constructs a non-RAG baseline prompt (parametric memory only)."""
    prompt = f"""أنت مستشار قانوني متخصص في القوانين الكويتية.

السؤال:
{question}

الإجابة (بالعربية الفصحى، مختصرة ودقيقة):
"""
    return prompt


def build_legal_analyst_json_prompt(
    question: str,
    evidences: List[Tuple[str, Dict[str, Any]]]
) -> str:
    """Constructs structured JSON legal analysis prompt for Kuwaiti Labor Law."""
    evidence_blocks = [
        format_evidence_block(chunk_text, meta, i + 1)
        for i, (chunk_text, meta) in enumerate(evidences)
    ]
    evidence_text = "\n\n".join(evidence_blocks)

    prompt = f"""أنت محلل قانوني متخصص في قانون العمل الكويتي.

مهمتك إصدار إجابة قانونية مسببة وبالاعتماد الحصري على المواد القانونية المسترجعة فقط.

قواعد ملزمة:
- استخدم النصوص المقدمة فقط.
- لا تستخدم أي معرفة خارجية أو من الذاكرة ولا تفترض مواد غير مذكورة.
- إذا لم تنطبق المواد → صرّح بعدم الانطباق ولا تستخدمها.
- ولا تكتب أي نص خارج JSON.
- ابدأ مباشرة بعلامة {{{{ وانتهِ بعلامة }}}} فقط.

خطوات التحليل داخلياً:
- تحديد المسألة القانونية محل النزاع
- استخراج القواعد والأحكام من المواد المسترجعة فقط
- حاول دائماً الاستفادة من المواد المتاحة وتحديد مدى انطباقها
- تطبيق القواعد على الوقائع وبيان وجه الانطباق
- إصدار النتيجة القانونية النهائية بصيغة جازمة

المواد القانونية:
{evidence_text}

السؤال:
{question}

أخرج JSON فقط مطابقاً تماماً لهذا الهيكل:
{{{{
"applicable": true,
"issue": "",
"rules": [],
"analysis": "",
"final_answer": "",
"referenced_articles": []
}}}}

تعليمات التعبئة:
- عند الانطباق → املأ جميع الحقول
- عند عدم الانطباق → applicable=false والبقية فارغة
"""
    return prompt
