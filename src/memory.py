"""Conversation memory: keeps recent turns verbatim, folds older ones into a summary.

A "turn" starts at a user message and includes every assistant/tool message
after it. We only ever drop whole turns, so a tool call is never separated
from its tool result (the API rejects that).
"""
import json


def estimate_tokens(messages, summary=""):
    """Rough size estimate: about 4 characters per token. Not exact."""
    return (len(summary) + sum(len(json.dumps(m, default=str)) for m in messages)) // 4


def split_turns(messages):
    turns = []
    for m in messages:
        if m["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(m)
    return turns


def extractive_summary(old_messages, previous=""):
    """No-LLM fallback: keep each old question and the final answer to it verbatim."""
    lines = [previous] if previous else []
    for m in old_messages:
        if m["role"] == "user":
            lines.append(f"Q: {m['content']}")
        elif m["role"] == "assistant" and not m.get("tool_calls") and m.get("content"):
            lines.append(f"A: {m['content']}")
    return "\n".join(lines)[-3000:]  # cap so the summary itself cannot grow forever


class ConversationMemory:
    def __init__(self, summarize=None, max_tokens=3000, keep_turns=2):
        self.messages = []
        self.summary = ""
        self.summarize = summarize or extractive_summary  # (old_messages, previous) -> str
        self.max_tokens = max_tokens
        self.keep_turns = keep_turns

    def add(self, message):
        self.messages.append(message)

    def build(self, system):
        """Messages to send to the model: system prompt (+ summary) then recent turns."""
        if self.summary:
            system = f"{system}\n\nSummary of the earlier conversation:\n{self.summary}"
        return [{"role": "system", "content": system}] + self.messages

    def compact(self):
        """Summarize old turns if over budget. Returns True if anything was dropped."""
        if estimate_tokens(self.messages, self.summary) <= self.max_tokens:
            return False
        turns = split_turns(self.messages)
        if len(turns) <= self.keep_turns:
            return False  # nothing old enough to drop
        old, recent = turns[: -self.keep_turns], turns[-self.keep_turns :]
        old_messages = [m for t in old for m in t]
        try:
            self.summary = self.summarize(old_messages, self.summary)
        except Exception:
            self.summary = extractive_summary(old_messages, self.summary)
        self.messages = [m for t in recent for m in t]
        return True
