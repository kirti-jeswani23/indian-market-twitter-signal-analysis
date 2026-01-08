"""
save_session.py
----------------
Launches a real Chrome browser and saves an authenticated
Twitter/X session for reuse in scraping.
"""

from playwright.sync_api import sync_playwright, TimeoutError

with sync_playwright() as p:
    browser = p.chromium.launch(
        channel="chrome",
        headless=False,
        args=["--disable-blink-features=AutomationControlled"]
    )

    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    )

    page = context.new_page()
    page.goto("https://twitter.com/login")

    print("👉 Log in manually in the opened Chrome window.")
    input("Press ENTER after login is complete...")

    # ✅ Wait for successful login indicators (NOT networkidle)
    try:
        page.wait_for_selector(
            '[data-testid="SideNav_NewTweet_Button"]',
            timeout=30000
        )
    except TimeoutError:
        # Fallback for layout changes
        page.wait_for_selector(
            '[data-testid="primaryColumn"]',
            timeout=30000
        )

    # Human-like pause
    page.wait_for_timeout(3000)

    context.storage_state(path="twitter_session.json")
    print("✅ Session saved to twitter_session.json")

    browser.close()
