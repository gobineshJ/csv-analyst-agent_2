"""Scripted stand-in for the Groq client, so the loop can be tested with no API key."""
from types import SimpleNamespace as NS


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
