"""Human-in-the-loop: ask a person before the agent runs a risky tool.

A request is {"tool": name, "args": {...}}. An approver returns a decision:
    {"action": "approve"}
    {"action": "edit", "args": {...}}        run with these arguments instead
    {"action": "reject", "reason": "..."}    don't run; the reason goes back to the model

The plain loop and MCP version call the approver directly and wait (the program blocks).
LangGraph pauses the graph with interrupt(), saves its state, and resumes later with the
decision, so the answer could come from another process, a web UI, or hours later.
"""

RISKY_TOOLS = {"run_pandas"}  # runs model-written code; the read-only tools never need approval


def auto_approve(request):
    return {"action": "approve"}


def console_approver(request):
    print(f"\nThe agent wants to run {request['tool']}:")
    for key, value in request["args"].items():
        print(f"--- {key} ---\n{value}")
    choice = input("Allow? [y]es / [n]o / [e]dit: ").strip().lower()
    if choice.startswith("y"):
        return {"action": "approve"}
    if choice.startswith("e") and "code" in request["args"]:
        print("Enter the new code, then an empty line:")
        lines = iter(input, "")
        return {"action": "edit", "args": {**request["args"], "code": "\n".join(lines)}}
    return {"action": "reject", "reason": input("Reason (optional): ").strip() or "not approved"}


def apply_decision(decision, args):
    """Return (args_to_run, None) or (None, rejection_message)."""
    action = decision.get("action")
    if action == "approve":
        return args, None
    if action == "edit":
        return decision["args"], None
    return None, f"The user rejected this call: {decision.get('reason', 'not approved')}. " \
                 "Do not retry the same call; explain or try another approach."
