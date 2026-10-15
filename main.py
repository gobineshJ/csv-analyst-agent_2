import sys
from dotenv import load_dotenv
load_dotenv()

from src.agent import run_agent, llm_summarizer
from src.memory import ConversationMemory
from src.tools import CsvTools

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_sales.csv"
tools = CsvTools(path)
memory = ConversationMemory(summarize=llm_summarizer())  # remembers across questions

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    print("\nANSWER:", run_agent(q, tools, memory=memory))
