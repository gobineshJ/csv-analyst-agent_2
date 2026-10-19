import sys
from dotenv import load_dotenv
load_dotenv()

from src.planner import build_planner, run_planner
from src.tools import CsvTools

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_sales.csv"
graph = build_planner(CsvTools(path))  # uses ChatGroq; needs GROQ_API_KEY. No memory between questions.

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    print("\nANSWER:", run_planner(q, graph))
