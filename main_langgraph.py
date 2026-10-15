import sys
from dotenv import load_dotenv
load_dotenv()

from src.langgraph_agent import build_graph, run_agent_langgraph
from src.tools import CsvTools

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_sales.csv"
graph = build_graph(CsvTools(path))  # uses ChatGroq; needs GROQ_API_KEY

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    print("\nANSWER:", run_agent_langgraph(q, graph))
