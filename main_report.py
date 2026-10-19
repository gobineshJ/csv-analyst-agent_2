"""Autonomous report agent: give it a goal and it works through the steps on its own.

    python main_report.py
    python main_report.py --goal "Compare the North and South regions" --max-calls 40
    python main_report.py   # again after a crash or budget stop: resumes from outputs/checkpoint.json
"""
import argparse
from dotenv import load_dotenv
load_dotenv()

from src.autonomous import run_autonomous
from src.hitl import auto_approve
from src.tools import CsvTools
from src.tracing import Budget

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv", nargs="?", default="data/sample_sales.csv")
parser.add_argument("--goal", default="Write a short sales report with 3 key insights and one chart.")
parser.add_argument("--max-calls", type=int, default=40, help="LLM call budget")
parser.add_argument("--max-seconds", type=int, default=600, help="time budget")
parser.add_argument("--yes", action="store_true", help="don't ask to approve the plan")
args = parser.parse_args()


def approve_plan(tasks):
    print("\nPlan:")
    for t in tasks:
        print(f"  {t.id} [{t.worker}] {t.task}")
    return input("Run this plan? [y/n]: ").strip().lower().startswith("y")


state = run_autonomous(args.goal, CsvTools(args.csv),
                       budget=Budget(max_llm_calls=args.max_calls, max_seconds=args.max_seconds),
                       approve_plan=None if args.yes else approve_plan,
                       approve=auto_approve)  # the plan was approved, so its code steps run unattended
print(f"\nStatus: {state['status']}")
if state.get("report_path"):
    print(f"Report: {state['report_path']}")
if state.get("totals"):
    t = state["totals"]
    print(f"Used {t['llm_calls']} LLM calls, {t['input_tokens'] + t['output_tokens']} tokens, ~${t['cost_usd']:.4f}")
