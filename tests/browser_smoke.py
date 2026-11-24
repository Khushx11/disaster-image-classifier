"""Exercise the recorded dashboard without local checkpoints."""
from pathlib import Path
import subprocess
import sys

from playwright.sync_api import expect, sync_playwright


root = Path(__file__).resolve().parents[1]
with subprocess.Popen(
    [sys.executable, str(root / "portfolio_demo.py"), "--port", "0"],
    stdout=subprocess.PIPE, text=True,
) as server:
    try:
        address = server.stdout.readline().strip().removeprefix("Open ")
        assert address.startswith("http://127.0.0.1:"), address
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="Disaster image informativeness")).to_be_visible()
            expect(page.get_by_text("0.8168", exact=True)).to_be_visible()
            expect(page.get_by_role("button", name="Run comparison")).to_have_count(0)
            page.get_by_label("Input condition").select_option("no_text")
            expect(page.get_by_label("Late fusion macro-F1")).to_have_js_property(
                "value", page.get_by_label("Image-only baseline macro-F1").evaluate("bar => bar.value")
            )
            assert not errors, errors
    finally:
        server.terminate()
