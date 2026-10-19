"""Web version of the agent. Open http://127.0.0.1:8000 after starting it."""
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from src.web_app import create_app

if __name__ == "__main__":
    # 127.0.0.1 only: the app has no login, so it must not be reachable from other machines.
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
