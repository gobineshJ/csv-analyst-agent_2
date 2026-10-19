from langchain_core.messages import AIMessage

from src.core.tools import CsvTools
from src.frameworks.langgraph_agent import build_graph, resume_langgraph, run_agent_langgraph, sqlite_checkpointer
from src.patterns.long_memory import (LongTermMemory, bm25_rank, build_system, fingerprint, load_chunks,
                                      run_agent_with_long_memory)
from tests.fakes import FakeChat, FakeClient, ai_call, text

CSV = "data/sample_sales.csv"
chunks = load_chunks("data/data_dictionary.md")


def test_retrieval_finds_the_relevant_definition():
    top = [chunks[i].splitlines()[0] for i in bm25_rank("How many big orders were there?", chunks, k=1)]
    assert top == ["## Big order"]
    assert bm25_rank("zebra giraffe", chunks) == []  # nothing relevant -> nothing added


def test_memory_persists_across_instances_and_is_per_dataset(tmp_path):
    db = tmp_path / "m.db"
    LongTermMemory(db).add("file-a", "qa", "Q: total? A: 5540")
    again = LongTermMemory(db)  # a new session / program run
    assert again.recall("file-a", "what was the total") == ["Q: total? A: 5540"]
    assert again.recall("file-b", "what was the total") == []  # a different file's notes are not used


def test_recall_combines_keyword_matches_and_recent_notes(tmp_path):
    store = LongTermMemory(tmp_path / "m.db")
    for note in ["Q: units in North? A: 100", "Q: grocery sales? A: 900", "Q: a? A: x", "Q: b? A: y"]:
        store.add("d", "qa", note)
    assert store.recall("d", "units North", k=1, recent=1) == ["Q: units in North? A: 100", "Q: b? A: y"]


def test_system_prompt_gets_definitions_and_notes(tmp_path):
    store = LongTermMemory(tmp_path / "m.db")
    store.remember("d", "The user is the South team manager.")
    system = build_system("How many big orders?", store, "d", chunks, verbose=False)
    assert "900 dollars or more" in system and "South team manager" in system


def test_answers_are_saved_for_next_session_unless_rejected(tmp_path):
    store, tools = LongTermMemory(tmp_path / "m.db"), CsvTools(CSV)
    run_agent_with_long_memory("total?", tools, store, chunks, client=FakeClient([text("5540")]), verbose=False)
    run_agent_with_long_memory("bad?", tools, store, chunks, client=FakeClient([text("9999")]), verbose=False,
                               store_if=lambda q, a: "9999" not in a)
    assert store.notes(fingerprint(CSV)) == ["Q: total? A: 5540"]

    client = FakeClient([text("ok")])  # next session sees the earlier answer
    run_agent_with_long_memory("what did you tell me before?", tools, store, chunks, client=client, verbose=False)
    assert "Q: total? A: 5540" in client.requests[0][0]["content"]


def test_langgraph_sqlite_checkpointer_survives_restart(tmp_path):
    db = str(tmp_path / "lg.db")
    llm = FakeChat(script=[AIMessage(content="first answer")], seen=[])
    run_agent_langgraph("first question", build_graph(CsvTools(CSV), llm, checkpointer=sqlite_checkpointer(db)),
                        thread_id="t", verbose=False)

    llm2 = FakeChat(script=[AIMessage(content="second")], seen=[])  # "restart": new graph, same file
    run_agent_langgraph("follow up", build_graph(CsvTools(CSV), llm2, checkpointer=sqlite_checkpointer(db)),
                        thread_id="t", verbose=False)
    assert "first question" in [m.content for m in llm2.seen[0]]


def test_paused_approval_can_be_resumed_after_restart(tmp_path):
    db = str(tmp_path / "lg.db")
    llm = FakeChat(script=[ai_call("run_pandas", {"code": "result = len(df)"})], seen=[])
    g1 = build_graph(CsvTools(CSV), llm, approve_code=True, checkpointer=sqlite_checkpointer(db))
    assert run_agent_langgraph("rows?", g1, thread_id="t", verbose=False).startswith("Waiting for approval")

    llm2 = FakeChat(script=[AIMessage(content="8 rows")], seen=[])
    g2 = build_graph(CsvTools(CSV), llm2, approve_code=True, checkpointer=sqlite_checkpointer(db))
    assert resume_langgraph(g2, {"action": "approve"}, thread_id="t", verbose=False) == "8 rows"
