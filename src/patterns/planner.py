"""Plan-and-execute, built as a LangGraph graph.

    START -> plan -> execute -> replan --(done)--> END
                        ^          |
                        +--(more)--+

- plan:    one LLM call writes a short numbered plan (no tools).
- execute: runs the NEXT step with the ordinary ReAct agent (langgraph_agent.build_graph).
- replan:  looks at the results so far and either answers or rewrites the remaining steps.

Compared with the ReAct loop, the model commits to a strategy up front and the plan is
visible, which helps multi-step questions. The cost: more LLM calls, and simple
questions get forced through the same machinery.
Planner outputs are validated with Pydantic (see structured.py for the same pattern).
"""
import json
import operator
import re
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from src.frameworks.langgraph_agent import build_graph, make_llm, run_agent_langgraph


class Plan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=5)


class Replan(BaseModel):
    done: bool
    answer: str = ""
    steps: list[str] = Field(default_factory=list, max_length=5)


class PlanState(TypedDict):
    question: str
    plan: list[str]
    past_steps: Annotated[list, operator.add]  # each node appends (step, result) pairs
    answer: str


PLANNER = ("You plan how to answer a question about a CSV file. Columns: {columns}. "
           "The executor can describe the data, aggregate, filter rows, sort, and run pandas code. "
           "Write the fewest steps needed (often 1 or 2). Each step must be a concrete instruction "
           "the executor can carry out with the tools. If the question cannot be answered from the "
           "data (e.g. predictions, columns that don't exist), use one step that checks the columns. "
           'Reply with JSON only: {{"steps": ["...", "..."]}}')
REPLANNER = ("You decide whether a question is answered. Question: {question}\n"
             "Original plan: {plan}\nCompleted steps and their results:\n{done}\n"
             "Remaining steps: {remaining}\n\n"
             "If the results answer the question (or show it cannot be answered), reply "
             '{{"done": true, "answer": "<final answer, using only numbers from the results>"}}. '
             'Otherwise reply {{"done": false, "steps": ["<the steps still needed>"]}}. JSON only.')


def ask_json(llm, prompt, model_cls, retries=1):
    """Call the model, parse its reply into model_cls, and send the error back if invalid."""
    messages = [SystemMessage(prompt), HumanMessage("Reply now with the JSON.")]
    for _ in range(retries + 1):
        text = llm.invoke(messages).content
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # models often add fences
        try:
            return model_cls.model_validate_json(cleaned)
        except ValidationError as e:
            messages += [AIMessage(text), HumanMessage(f"Invalid: {e}. Reply with corrected JSON only.")]
    raise ValueError(f"model did not return valid {model_cls.__name__}: {text[:200]}")


def build_planner(csv, llm=None, max_steps=6, verbose=True):
    llm = llm or make_llm()
    executor = build_graph(csv, llm)  # the ReAct agent; each step runs on a fresh thread
    columns = ", ".join(f"{c} ({t})" for c, t in csv.describe_data()["columns"].items())

    def plan(state: PlanState):
        steps = ask_json(llm, PLANNER.format(columns=columns) + f"\nQuestion: {state['question']}", Plan).steps
        if verbose:
            print("PLAN   " + " | ".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
        return {"plan": steps}

    def execute(state: PlanState):
        step = state["plan"][0]
        done = "\n".join(f"- {s}: {r}" for s, r in state["past_steps"]) or "(none)"
        task = (f"Overall question: {state['question']}\nAlready done:\n{done}\n\n"
                f"Do only this step and report the result with its numbers: {step}")
        if verbose:
            print(f"STEP   {step}")
        result = run_agent_langgraph(task, executor, thread_id=str(uuid.uuid4()), verbose=verbose)
        return {"past_steps": [(step, result)], "plan": state["plan"][1:]}

    def replan(state: PlanState):
        if len(state["past_steps"]) >= max_steps:
            return {"answer": "Stopped: hit the plan step limit. Last result: " + state["past_steps"][-1][1]}
        done = "\n".join(f"- {s}: {r}" for s, r in state["past_steps"])
        decision = ask_json(llm, REPLANNER.format(question=state["question"], done=done,
                                                  plan=json.dumps([s for s, _ in state["past_steps"]] + state["plan"]),
                                                  remaining=json.dumps(state["plan"])), Replan)
        if decision.done or not decision.steps:
            return {"answer": decision.answer or state["past_steps"][-1][1]}
        if verbose:
            print("REPLAN " + " | ".join(decision.steps))
        return {"plan": decision.steps}

    graph = StateGraph(PlanState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("replan", replan)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "replan")
    graph.add_conditional_edges("replan", lambda s: END if s.get("answer") else "execute")
    return graph.compile()


def run_planner(question, graph):
    return graph.invoke({"question": question, "past_steps": [], "answer": ""},
                        {"recursion_limit": 50})["answer"]
