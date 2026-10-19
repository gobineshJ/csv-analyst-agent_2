"""Structured output: turn the agent's free-text answer into a validated object.

Free text is fine for a person but hard for a program to use. Here the model is asked
for JSON matching FinalAnswer; Pydantic checks it; if it is invalid, the validation error
is sent back and the model tries again. This validate-and-retry loop is the core pattern
for any agent whose output feeds other code.

The schema checks shape (types, required fields) AND rules a JSON schema can't express
(see check_consistency): validation is where you encode what "a correct answer" means.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.core.agent import MODEL, SYSTEM, get_client


class FinalAnswer(BaseModel):
    answerable: bool = Field(description="false if the tools could not answer the question")
    answer: str = Field(min_length=1, description="the answer in one or two sentences")
    value: float | None = Field(None, description="the main number in the answer, if there is one")
    unit: str | None = Field(None, description="what the value measures, e.g. 'sales' or 'units'")
    tools_used: list[str] = Field(default_factory=list, description="names of the tools the answer relies on")
    confidence: Literal["high", "medium", "low"]

    @model_validator(mode="after")
    def check_consistency(self):
        if not self.answerable and self.value is not None:
            raise ValueError("an unanswerable question cannot have a value; set value to null")
        if self.answerable and not self.tools_used:
            raise ValueError("an answer must name the tools it relies on; set answerable to false "
                             "if no tool was used")
        return self


FINALIZE_PROMPT = ("Convert your last answer into JSON matching this schema, using only facts from "
                   "this conversation. Reply with the JSON object only.\n{schema}")


def ask_json(client, messages, model_cls, retries=2, verbose=True, max_tokens=600, context=None):
    """Call the model in JSON mode and validate into model_cls, sending errors back to retry.
    context is passed to Pydantic validators, for rules that depend on the request.
    Returns (instance or None, attempts). Used by finalize, multi_agent and autonomous."""
    messages = list(messages)  # temporary: the JSON exchange is never stored in a memory
    for attempt in range(1, retries + 2):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens,
                                                  response_format={"type": "json_object"})
            text = resp.choices[0].message.content or ""
        except Exception as e:  # Groq checks JSON mode itself and may answer 400 json_validate_failed
            if "json_validate_failed" not in str(e):
                raise
            text, error = "", "the reply was not valid JSON"
        else:
            try:
                return model_cls.model_validate_json(text, context=context), attempt
            except ValidationError as e:
                error = str(e)
        if verbose:
            print(f"SCHEMA attempt {attempt} invalid: {error.splitlines()[0][:120]}")
        messages += [{"role": "assistant", "content": text or "(invalid)"},
                     {"role": "user", "content": f"That JSON is invalid:\n{error}\nReply with corrected JSON only."}]
    return None, retries + 1


def finalize(memory, client=None, retries=2, verbose=True):
    """Ask for the last answer as FinalAnswer JSON. Returns (FinalAnswer or None, attempts)."""
    schema = json.dumps(FinalAnswer.model_json_schema())
    messages = memory.build(SYSTEM) + [{"role": "user", "content": FINALIZE_PROMPT.format(schema=schema)}]
    return ask_json(client or get_client(), messages, FinalAnswer, retries, verbose)


def run_agent_structured(question, tools, memory=None, client=None, verbose=True, **kwargs):
    """Run the normal agent, then convert its answer. Returns (FinalAnswer or None, text answer)."""
    from src.core.agent import run_agent
    from src.core.memory import ConversationMemory

    memory = memory or ConversationMemory()
    text = run_agent(question, tools, memory=memory, client=client, verbose=verbose, **kwargs)
    result, attempts = finalize(memory, client, verbose=verbose)
    if verbose and result:
        print(f"STRUCTURED {result.model_dump_json()} (attempts: {attempts})")
    return result, text
