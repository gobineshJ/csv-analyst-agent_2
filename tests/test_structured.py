import json

import pytest
from pydantic import ValidationError

from src.memory import ConversationMemory
from src.structured import FinalAnswer, finalize, run_agent_structured
from src.tools import CsvTools
from tests.fakes import FakeClient, call, text

GOOD = {"answerable": True, "answer": "Total sales are 5540.", "value": 5540, "unit": "sales",
        "tools_used": ["aggregate"], "confidence": "high"}


def test_schema_rules_beyond_types():
    FinalAnswer(**GOOD)
    with pytest.raises(ValidationError):
        FinalAnswer(**{**GOOD, "answerable": False})  # unanswerable but has a value
    with pytest.raises(ValidationError):
        FinalAnswer(**{**GOOD, "tools_used": []})
    with pytest.raises(ValidationError):
        FinalAnswer(**{**GOOD, "confidence": "very"})


def test_invalid_json_is_sent_back_and_retried():
    memory = ConversationMemory()
    memory.add({"role": "user", "content": "total?"})
    client = FakeClient([text("not json"), text(json.dumps({**GOOD, "confidence": "sure"})), text(json.dumps(GOOD))])
    result, attempts = finalize(memory, client, verbose=False)
    assert result.value == 5540 and attempts == 3
    assert "invalid" in client.requests[1][-1]["content"]
    assert len(memory.messages) == 1  # the JSON exchange is not stored


def test_gives_up_after_retries():
    client = FakeClient([text("nope")] * 3)
    assert finalize(ConversationMemory(), client, retries=2, verbose=False) == (None, 3)


def test_run_agent_structured_end_to_end():
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum"}'), text("5540"), text(json.dumps(GOOD))])
    result, answer = run_agent_structured("total?", CsvTools("data/sample_sales.csv"), client=client, verbose=False)
    assert answer == "5540" and result.tools_used == ["aggregate"]
