"""Scripted stand-ins for the models, so every test runs with no API key and no network.

- FakeClient replaces the Groq client (plain loop, MCP, reflection, structured, multi-agent).
- FakeChat replaces the LangChain chat model (LangGraph, planner, web app).
"""
from types import SimpleNamespace as NS

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# --- Groq-style client ------------------------------------------------------------------

def text(content):
    return NS(choices=[NS(message=NS(content=content, tool_calls=None))])


def call(name, args_json, call_id="c1"):
    tc = NS(id=call_id, function=NS(name=name, arguments=args_json))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []  # messages sent on each call, for assertions
        self.kwargs = []    # everything else sent (tools, response_format, ...)
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(list(kw["messages"]))
        self.kwargs.append(kw)
        response = self.responses.pop(0)
        if isinstance(response, Exception):  # script an API error
            raise response
        return response


# --- LangChain chat model ---------------------------------------------------------------

class FakeChat(BaseChatModel):
    script: list
    seen: list = []

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def ai_call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
