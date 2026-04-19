import re
from openai import OpenAI
from config import BASE_URL, API_KEY, MODEL_NAME


_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def is_answer_suspicious(answer: str, expected_type: str) -> bool:
    """
    Lightweight heuristic: should we trigger a query rewrite for this answer?
    Returns True to suggest a rewrite; not a hard rejection.
    """
    if not answer or answer.strip().lower() in ("[none]", "answer not found", ""):
        return True

    a = answer.strip().lower()
    words = a.split()

    # Overly long answers are likely hallucinated narratives
    if len(words) > 20:
        return True

    if expected_type == "date":
        # Expect at least one 4-digit year or common date word
        if not re.search(r'\b\d{4}\b|\b\d{1,2}(st|nd|rd|th)?\b', a):
            return True

    elif expected_type == "number":
        if not re.search(r'\d', a):
            return True

    elif expected_type == "person":
        # Pure numeric string makes no sense as a person name
        if re.fullmatch(r'[\d\s\-,.]+', a):
            return True

    elif expected_type == "org":
        if re.fullmatch(r'[\d\s\-,.]+', a):
            return True

    return False


def check_fallback_supported(question: str, candidate: str, retrieved_docs: str) -> bool:
    """
    Check whether retrieved docs clearly contradict the candidate answer.
    Returns True (supported/allowed) if docs do NOT contradict the candidate.
    Fails open (returns True) on any error so we don't accidentally block a good answer.
    """
    docs_snippet = retrieved_docs[:3000] if retrieved_docs else ""
    prompt = (
        f"Documents:\n{docs_snippet}\n\n"
        f"Question: {question}\n"
        f"Candidate answer: {candidate}\n\n"
        "Do the documents clearly contradict or provide strong evidence against this candidate answer? "
        "Reply with only 'yes' (contradicted) or 'no' (not contradicted)."
    )
    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip().lower()
        # "yes" = contradicted = block; "no" = not contradicted = allow
        return not reply.startswith("y")
    except Exception:
        # Fail open: don't block the answer if the support check itself fails
        return True
