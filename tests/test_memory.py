from src.memory import ConversationMemory, split_turns, extractive_summary


def turn(q, a, tool=True):
    msgs = [{"role": "user", "content": q}]
    if tool:
        msgs += [{"role": "assistant", "content": None,
                  "tool_calls": [{"id": "x", "type": "function",
                                  "function": {"name": "aggregate", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "x", "content": "{\"result\": 1}" + " " * 400}]
    msgs.append({"role": "assistant", "content": a})
    return msgs


def fill(mem, n):
    for i in range(n):
        for m in turn(f"question {i}", f"answer {i}"):
            mem.add(m)


def test_no_compaction_under_budget():
    mem = ConversationMemory(max_tokens=10_000)
    fill(mem, 3)
    assert mem.compact() is False and mem.summary == ""


def test_compaction_keeps_recent_turns_and_summarizes_old():
    mem = ConversationMemory(max_tokens=200, keep_turns=2)
    fill(mem, 5)
    assert mem.compact() is True
    users = [m["content"] for m in mem.messages if m["role"] == "user"]
    assert users == ["question 3", "question 4"]
    assert "question 0" in mem.summary and "answer 2" in mem.summary


def test_never_orphans_a_tool_message():
    mem = ConversationMemory(max_tokens=200, keep_turns=2)
    fill(mem, 5)
    mem.compact()
    assert mem.messages[0]["role"] == "user"  # recent history starts at a turn boundary
    ids_called = {tc["id"] for m in mem.messages for tc in m.get("tool_calls", [])}
    ids_returned = {m["tool_call_id"] for m in mem.messages if m["role"] == "tool"}
    assert ids_returned <= ids_called


def test_too_few_turns_means_nothing_dropped():
    mem = ConversationMemory(max_tokens=10, keep_turns=2)
    fill(mem, 2)
    assert mem.compact() is False and len(split_turns(mem.messages)) == 2


def test_summary_goes_into_system_prompt():
    mem = ConversationMemory(max_tokens=200, keep_turns=1)
    fill(mem, 3)
    mem.compact()
    built = mem.build("SYS")
    assert built[0]["role"] == "system" and "question 0" in built[0]["content"]
    assert sum(m["role"] == "system" for m in built) == 1


def test_failing_summarizer_falls_back():
    def boom(old, prev):
        raise RuntimeError("api down")
    mem = ConversationMemory(summarize=boom, max_tokens=200, keep_turns=1)
    fill(mem, 3)
    assert mem.compact() is True
    assert "question 0" in mem.summary


def test_summary_is_capped():
    old = [m for i in range(400) for m in turn(f"question number {i}", f"answer {i}", tool=False)]
    assert len(extractive_summary(old)) <= 3000
