import json
from groq import Groq
from src.memory import ConversationMemory
from src.schemas import TOOLS

MODEL = "openai/gpt-oss-120b"  # check Groq's docs for current tool-capable models
ALLOWED = {"describe_data", "aggregate", "filter_rows", "top_n"}
SYSTEM = ("You analyse a CSV using the provided tools. Only state numbers that tools returned. "
          "If the tools cannot answer the question, say so instead of guessing.")
SUMMARY_PROMPT = ("Summarize this conversation about a CSV file in under 150 words. "
                  "Keep every column name, number and conclusion exactly as stated. "
                  "Do not add anything that is not in the conversation.")

_client = None


def get_client():
    """Created on first use so importing this module does not need an API key."""
    global _client
    if _client is None:
        _client = Groq()
    return _client


def llm_summarizer(client=None):
    """Summarizer for ConversationMemory that asks the model to compress old turns."""
    def summarize(old_messages, previous):
        text = "\n".join(f"{m['role']}: {m.get('content') or m.get('tool_calls')}" for m in old_messages)
        if previous:
            text = f"Earlier summary:\n{previous}\n\nNew messages:\n{text}"
        resp = (client or get_client()).chat.completions.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "system", "content": SUMMARY_PROMPT},
                      {"role": "user", "content": text}])
        return resp.choices[0].message.content
    return summarize


def run_agent(question, tools, max_steps=8, verbose=True, memory=None, client=None):
    client = client or get_client()
    memory = memory or ConversationMemory()  # fresh memory = old single-question behaviour
    memory.add({"role": "user", "content": question})
    answer = "Stopped: hit the step limit."
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=MODEL, messages=memory.build(SYSTEM),
                                              tools=TOOLS, max_tokens=1024)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            break

        memory.add({"role": "assistant", "content": msg.content,
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"CALL   {name} {args}")
                if name not in ALLOWED:
                    raise ValueError(f"unknown tool {name}")
                out = getattr(tools, name)(**args)
            except Exception as e:
                out = {"error": str(e)}
            if verbose:
                print(f"RESULT {str(out)[:200]}")
            memory.add({"role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(out, default=str)})
    memory.add({"role": "assistant", "content": answer})
    if memory.compact() and verbose:
        print("[memory] older turns summarized")
    return answer
