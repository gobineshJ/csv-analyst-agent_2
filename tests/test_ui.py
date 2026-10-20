"""End-to-end test of the web page in a real browser (Edge via Playwright), with the fake models.

Skipped automatically when Playwright or Microsoft Edge isn't installed:
    pip install playwright        (no browser download needed: it drives the installed Edge)
"""
import json
import socket
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from tests.fakes import FakeChat, FakeClient, ai_call, text

sync_api = pytest.importorskip("playwright.sync_api")
DATA = Path(__file__).resolve().parent.parent / "data"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    import uvicorn

    from src.web.app import create_app

    monkeypatch.chdir(tmp_path)
    llm = FakeChat(seen=[], script=[
        ai_call("aggregate", {"column": "sales", "how": "sum"}),          # question 1
        AIMessage(content="Total sales are **5540**. <img src=x onerror=alert(1)>"),
        ai_call("run_pandas", {"code": "result = len(df)"}, "c2"),       # question 2: needs approval
        AIMessage(content="| Rows |\n|---|\n| 8 |"),
    ])
    plan = {"tasks": [{"id": f"t{i}", "worker": "analyst", "task": f"Task {i}"} for i in (1, 2, 3)]}
    client = FakeClient([text(json.dumps(plan)), text("Total 5540."), text("North 3940."), text("South 1600."),
                         text("# Sales report\n\n1. Total sales were **5540**.")])
    app = create_app(data_dir=DATA, db_path=str(tmp_path / "web.db"), llm=llm, client=client)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


@pytest.fixture
def page(server):
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:
            pytest.skip("Microsoft Edge is not installed")
        page = browser.new_page()
        page.set_default_timeout(15000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server)
        yield page
        browser.close()
        assert not errors, errors  # no JavaScript errors anywhere in the run


def test_full_journey(page, tmp_path):
    expect = sync_api.expect
    expect(page.locator(".welcome h1")).to_have_text("Ask your data a question")
    expect(page.locator("#dataset")).to_have_value("sample_sales.csv")

    # Ask: tool call is shown, answer is rendered as Markdown, and HTML in the answer stays text.
    page.fill("#question", "What are total sales?")
    page.keyboard.press("Enter")
    answer = page.locator(".answer").first
    expect(answer).to_contain_text("5540")
    expect(answer.locator("strong")).to_have_text("5540")
    expect(answer).to_contain_text("<img src=x onerror=alert(1)>")
    assert page.locator(".answer img").count() == 0
    expect(page.locator(".activity summary").first).to_contain_text("Used 1 tool call")

    # Code needs approval: the composer is locked until the human decides.
    page.fill("#question", "How many rows? Use code.")
    page.keyboard.press("Enter")
    expect(page.locator(".approval textarea")).to_have_value("result = len(df)")
    expect(page.locator("#question")).to_be_disabled()
    page.click(".approval button.primary")
    expect(page.locator(".decision")).to_contain_text("You approved the code")
    expect(page.locator(".answer table td")).to_have_text("8")
    expect(page.locator("#question")).to_be_enabled()

    # Reload: the conversation comes back from the server.
    page.reload()
    expect(page.locator(".answer")).to_have_count(2)
    expect(page.locator(".session.active .name")).to_have_text("What are total sales?")

    # Preview and upload.
    page.click("#previewBtn")
    expect(page.locator("#previewTitle")).to_have_text("sample_sales.csv · 8 rows")
    expect(page.locator("#previewBody tbody tr")).to_have_count(8)
    page.click("#closePreview")
    csv = tmp_path / "my data.csv"
    csv.write_text("city,visits\nParis,3\nRome,5\n")
    page.set_input_files("#fileInput", str(csv))
    expect(page.locator("#toast")).to_contain_text("Uploaded my_data.csv: 2 rows, 2 columns")
    expect(page.locator("#dataset")).to_have_value("my_data.csv")

    # Report tab: plan, progress, rendered report.
    page.select_option("#dataset", "sample_sales.csv")
    page.click("#tabReport")
    page.click("#runReport")
    expect(page.locator(".plan li")).to_have_count(3)
    expect(page.locator("#reportDoc h2")).to_have_text("Sales report")
    expect(page.locator(".plan li.done")).to_have_count(3)

    # Delete the conversation.
    page.click("#tabChat")
    page.once("dialog", lambda d: d.accept())
    page.locator(".session.active .del").click()
    expect(page.locator(".empty-list")).to_have_text("No conversations yet.")
    expect(page.locator(".welcome")).to_be_visible()


def test_phone_layout(page, server):
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(server)
    sync_api.expect(page.locator("#openSidebar")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth") <= 390  # no sideways scrolling
    page.click("#openSidebar")
    sync_api.expect(page.locator("#newChat")).to_be_in_viewport()
