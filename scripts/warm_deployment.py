"""Keeps this project's Hugging Face Space reachable.

A Space on free CPU hardware is suspended after a period without traffic.
The Gradio app is a JavaScript client, so a plain HTTP request gets the static
shell back and starts nothing. This loads the Space in a headless browser and
clicks the restart control if the Space has already been suspended.

Doubles as a deployment smoke test: a non-zero exit means the live Space is
unreachable.
"""

from playwright.sync_api import sync_playwright

APP_URL = "https://themegalodon55681-neurolens.hf.space"

WAKE_TEXTS = [
    "restart this space",
    "get this space back up",
    "wake",
]

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page()
    try:
        page.goto(APP_URL, wait_until="networkidle", timeout=120_000)
        clicked = False
        for text in WAKE_TEXTS:
            control = page.get_by_text(text, exact=False)
            if control.count():
                control.first.click()
                page.wait_for_timeout(60_000)
                clicked = True
                break
        print(f"{'WOKE' if clicked else 'OK  '} {APP_URL}")
    finally:
        page.close()
        browser.close()
