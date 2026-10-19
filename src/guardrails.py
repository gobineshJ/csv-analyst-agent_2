"""Guardrails against prompt injection through the data.

Tool results go straight into the model's context. If a CSV cell says "ignore previous
instructions and ...", the model may obey it, because to the model it is just more text.
Two defences are used together:

1. The system prompt says tool results are data, never instructions (see agent.SYSTEM).
2. Every string in a tool result is scanned, and text that looks like instructions to an
   AI is replaced before the model sees it.

Pattern matching is easy to get around (rephrasing, other languages, encodings), so this
reduces risk but does not remove it. The real protection is limiting what the agent can
do: this agent can only read the CSV, so a successful injection can make it lie but
cannot make it delete files or send data anywhere.
"""
import re

INJECTION = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|prompts?|rules)"
    r"|disregard (?:all |the )?(?:previous|prior|above|system)"
    r"|(?:system|developer) (?:prompt|message|note|instruction)"
    r"|you are now|new instructions|act as (?:an?|the) "
    r"|do not (?:tell|inform) the user"
    r"|<\s*/?\s*(?:system|assistant|instructions?)\s*>",
    re.I)
REPLACEMENT = "[removed by guardrail: this text looked like instructions to the AI]"


def looks_like_injection(text):
    return bool(INJECTION.search(text))


def sanitize(value):
    """Return a copy of a tool result with suspicious strings replaced. Also reports what was hit."""
    hits = []

    def clean(v):
        if isinstance(v, str) and looks_like_injection(v):
            hits.append(v[:80])
            return REPLACEMENT
        if isinstance(v, dict):
            return {clean(k): clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    cleaned = clean(value)
    if hits and isinstance(cleaned, dict):
        cleaned["guardrail"] = f"{len(hits)} value(s) removed as possible prompt injection"
    return cleaned
