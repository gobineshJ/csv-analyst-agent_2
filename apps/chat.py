"""Plain-loop agent chat, with a flag for every stage.  python -m apps.chat --help"""
import argparse
from dotenv import load_dotenv
load_dotenv()

from src.core.agent import llm_summarizer, run_agent, get_client, MODEL
from src.core.memory import ConversationMemory
from src.core.schemas import CODE_TOOLS, FIXED_TOOLS, TOOLS
from src.core.tools import CsvTools
from src.patterns.hitl import auto_approve, console_approver
from src.patterns.long_memory import LongTermMemory, fingerprint, load_chunks, run_agent_with_long_memory
from src.patterns.multi_agent import run_multi_agent
from src.patterns.reflection import llm_critic, make_reviewer
from src.patterns.structured import finalize
from src.patterns.tracing import TracedClient, TracedTools, Tracer

parser = argparse.ArgumentParser(description="Plain-loop CSV agent")
parser.add_argument("csv", nargs="?", default="data/sample_sales.csv")
parser.add_argument("--tools", choices=["all", "fixed", "code"], default="all",
                    help="fixed = the 4 pandas tools, code = run_pandas only, all = both")
parser.add_argument("--reflect", action="store_true", help="review each answer before showing it")
parser.add_argument("--critic", action="store_true", help="with --reflect, also ask an LLM critic")
parser.add_argument("--structured", action="store_true", help="also print the answer as validated JSON")
parser.add_argument("--auto-approve", action="store_true", help="run model-written code without asking")
parser.add_argument("--no-guard", action="store_true", help="turn off the prompt-injection guardrail")
parser.add_argument("--trace", action="store_true", help="log every LLM/tool call to runtime/traces/ and print totals")
parser.add_argument("--long-memory", action="store_true",
                    help="remember across runs (runtime/memory.db) and use data/data_dictionary.md")
parser.add_argument("--multi", action="store_true", help="orchestrator + analyst/chart specialists")
args = parser.parse_args()

tracer = Tracer() if args.trace else None
client = TracedClient(get_client(), tracer) if tracer else get_client()
tools = CsvTools(args.csv, guard=not args.no_guard)
tools = TracedTools(tools, tracer) if tracer else tools
memory = ConversationMemory(summarize=llm_summarizer(client))  # remembers across questions
schemas = {"all": TOOLS, "fixed": FIXED_TOOLS, "code": CODE_TOOLS}[args.tools]
review = make_reviewer(llm_critic(client, MODEL) if args.critic else None) if args.reflect else None
approve = auto_approve if args.auto_approve else console_approver
store = LongTermMemory() if args.long_memory else None
chunks = load_chunks("data/data_dictionary.md") if args.long_memory else None

while True:
    q = input("\nAsk a question (or 'quit', or 'remember: <fact>'): ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    if q.lower().startswith("remember:") and store:
        store.remember(fingerprint(args.csv), q.split(":", 1)[1].strip())
        print("Saved for future sessions.")
        continue
    if args.multi:
        answer, _ = run_multi_agent(q, tools, client=client, approve=approve)
    elif store:
        answer = run_agent_with_long_memory(q, tools, store=store, chunks=chunks, memory=memory, client=client,
                                            schemas=schemas, approve=approve, review=review)
    else:
        answer = run_agent(q, tools, memory=memory, client=client, schemas=schemas, approve=approve, review=review)
    print("\nANSWER:", answer)
    if args.structured:
        result, attempts = finalize(memory, client)
        print("JSON:", result.model_dump_json(indent=2) if result else f"invalid after {attempts} attempts")
    if tracer:
        t = tracer.totals()
        print(f"[trace] {t['llm_calls']} LLM calls, {t['input_tokens']}+{t['output_tokens']} tokens, "
              f"{t['tool_calls']} tool calls, ~${t['cost_usd']:.4f} so far. Details: {tracer.path}")
