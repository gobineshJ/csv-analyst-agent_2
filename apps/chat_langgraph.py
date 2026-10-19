"""The agent as a LangGraph graph; pauses for approval before running code.  python -m apps.chat_langgraph --help"""
import argparse
from dotenv import load_dotenv
load_dotenv()

from src.core.tools import CsvTools
from src.frameworks.langgraph_agent import build_graph, run_agent_langgraph, sqlite_checkpointer
from src.patterns.hitl import console_approver

parser = argparse.ArgumentParser(description="LangGraph CSV agent")
parser.add_argument("csv", nargs="?", default="data/sample_sales.csv")
parser.add_argument("--db", help="keep conversation memory in this SQLite file across runs, e.g. runtime/langgraph.db")
parser.add_argument("--thread", default="default", help="conversation id; reuse it to continue a conversation")
args = parser.parse_args()

# uses ChatGroq; needs GROQ_API_KEY. run_pandas pauses the graph until you approve it.
graph = build_graph(CsvTools(args.csv), approve_code=True,
                    checkpointer=sqlite_checkpointer(args.db) if args.db else None)

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    print("\nANSWER:", run_agent_langgraph(q, graph, thread_id=args.thread, ask=console_approver))
