import json
from types import SimpleNamespace as NS

from src.agent import REVIEW_PREFIX, run_agent
from src.reflection import llm_critic, make_reviewer, ungrounded_numbers
from src.tools import CsvTools
from tests.fakes import FakeClient, call, text

tools = CsvTools("data/sample_sales.csv")


def test_ungrounded_numbers():
    evidence = ['{"result": 5540.0}']
    assert ungrounded_numbers("Total sales are 5,540.", evidence) == []
    assert ungrounded_numbers("Total sales are 6,000.", evidence) == [6000]
    assert ungrounded_numbers("There are 3 categories.", evidence) == []  # small counts ignored
    assert ungrounded_numbers("Sales since 2025 were 5540", evidence, "since 2025?") == []


def test_review_sends_problem_back_and_model_corrects_itself():
    client = FakeClient([call("aggregate", '{"column":"sales","how":"sum"}'),
                         text("Total sales are 6000."),          # made-up number
                         text("Total sales are 5540.")])
    answer = run_agent("total?", tools, client=client, verbose=False, review=make_reviewer())
    assert answer == "Total sales are 5540."
    feedback = client.requests[2][-1]
    assert feedback["role"] == "user" and feedback["content"].startswith(REVIEW_PREFIX) and "6000" in feedback["content"]


def test_review_happens_at_most_max_reviews_times():
    client = FakeClient([text("It is 7777."), text("It is 8888.")])
    assert run_agent("q", tools, client=client, verbose=False, review=make_reviewer()) == "It is 8888."
    assert len(client.requests) == 2


def test_llm_critic_runs_only_when_grounding_passes():
    critic_client = FakeClient([NS(choices=[NS(message=NS(content=json.dumps(
        {"ok": False, "problem": "answers a different question"}), tool_calls=None))])])
    review = make_reviewer(llm_critic(critic_client, "m"))
    assert review("q", "It is 5540.", ['{"result": 5540}']) == "answers a different question"
    assert "not in any tool result" in review("q", "It is 1.5.", ['{"result": 5540}'])
    assert len(critic_client.requests) == 1  # the grounding failure didn't need the critic


def test_broken_critic_fails_open():
    # Seen live: Groq returned 400 json_validate_failed from the critic call.
    review = make_reviewer(llm_critic(FakeClient([RuntimeError("json_validate_failed")]), "m"))
    assert review("q", "It is 5540.", ['{"result": 5540}']) is None
