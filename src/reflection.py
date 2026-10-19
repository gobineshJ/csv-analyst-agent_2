"""Reflection: check the agent's draft answer before the user sees it.

Two reviewers, cheapest first:
1. Grounding check (no LLM): every number in the answer must appear in a tool result
   or in the question. Catches made-up numbers and numbers the model calculated in its
   head (the system prompt says to use a tool for those).
2. LLM critic (optional): a second model call judges whether the answer addresses the
   question and is supported by the tool results. Catches problems with no number in them,
   but costs a call and can itself be wrong.

If a reviewer finds a problem, the agent loop sends it back to the model as feedback
and lets it try again (see run_agent's `review` argument).
"""
import json
import re

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CRITIC_PROMPT = (
    "You review answers from a data-analysis agent. You get the question, the tool results the "
    "agent saw, and its answer. Reply in JSON only: {\"ok\": true} if the answer addresses the "
    "question and every claim in it is supported by the tool results (or it correctly says the "
    "tools cannot answer), otherwise {\"ok\": false, \"problem\": \"<one sentence>\"}.")


def numbers(text):
    out = set()
    for n in NUMBER.findall(text):
        try:
            out.add(float(n.replace(",", "")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(answer, evidence, question=""):
    """Numbers in the answer that appear in no tool result and not in the question.

    Integers below 10 are ignored: they are usually counts or list positions ("3 categories",
    "top 5") and flagging them causes more false alarms than it catches.
    """
    known = numbers(question)
    for e in evidence:
        known |= numbers(e)

    def is_known(n):
        return any(abs(n - k) <= max(0.01, abs(k) * 0.005) for k in known)
    return sorted(n for n in numbers(answer) if not (n < 10 and n.is_integer()) and not is_known(n))


def llm_critic(client, model):
    def critic(question, answer, evidence):
        content = (f"Question: {question}\n\nTool results:\n" + "\n".join(evidence)[-6000:] +
                   f"\n\nAnswer: {answer}")
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=300, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": CRITIC_PROMPT}, {"role": "user", "content": content}])
            verdict = json.loads(resp.choices[0].message.content)
        except Exception:
            return None  # fail open: a broken critic (bad JSON, API error) should not block an answer
        return None if verdict.get("ok") else verdict.get("problem", "the reviewer rejected the answer")
    return critic


def make_reviewer(critic=None):
    """Return review(question, answer, evidence) -> problem text, or None if the answer is fine."""
    def review(question, answer, evidence):
        if bad := ungrounded_numbers(answer, evidence, question):
            shown = ", ".join(f"{n:g}" for n in bad)
            return (f"These numbers are not in any tool result: {shown}. Do not calculate in your head: "
                    "get every number from a tool (run_pandas can compute derived values), "
                    "or remove it from the answer.")
        return critic(question, answer, evidence) if critic else None
    return review
