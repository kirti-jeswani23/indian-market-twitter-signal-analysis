"""
twitter_scraper.py
------------------
Scrapes live tweets for Indian stock market hashtags
using Playwright with authenticated session persistence.
"""

import re
import json
import logging
import random
import unicodedata
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timedelta, timezone

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError

# ---------------- CONFIG ---------------- #
HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
TARGET_TWEETS = 2000
MAX_SCROLLS_PER_TAG = 10
OUTPUT_PARQUET = "data/indian_market_tweets.parquet"
OUTPUT_CSV = "data/indian_market_tweets.csv"
CHUNK_SIZE = 200

# ---------------- SETUP ---------------- #
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)

NOW_UTC = datetime.now(timezone.utc)
CUTOFF_TIME = NOW_UTC - timedelta(hours=24)

# ---------------- UTILS ---------------- #
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_count(locator) -> int:
    try:
        txt = locator.inner_text().upper().replace(",", "")
        m = re.match(r"([\d\.]+)([KM]?)", txt)
        if not m:
            return 0
        val = float(m.group(1))
        return int(val * 1_000_000 if m.group(2) == "M"
                   else val * 1_000 if m.group(2) == "K"
                   else val)
    except:
        return 0


def flush_to_parquet(rows):
    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["permalink"], inplace=True)

    if Path(OUTPUT_PARQUET).exists():
        old = pd.read_parquet(OUTPUT_PARQUET)
        df = pd.concat([old, df]).drop_duplicates(subset=["permalink"])

    df.to_parquet(
        OUTPUT_PARQUET,
        engine="pyarrow",
        compression="snappy",
        index=False
    )
    # Save CSV
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    logging.info(f"Flushed {len(df)} total records (Parquet + CSV)")



# ---------------- SCRAPER ---------------- #
def scrape_tweets():
    all_rows = []
    seen_links: set[str] = set()
    buffer: list[dict] = []


    with sync_playwright() as p:
        storage = json.load(open("twitter_session.json"))

        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(storage_state=storage)
        page = context.new_page()

        # ✅ Verify login
        page.goto("https://x.com/home", timeout=60000)
        page.wait_for_selector('[data-testid="primaryColumn"]', timeout=30000)

        REPEAT_THRESHOLD = 5  # Stop after 5 scrolls with no new unique tweets

        for tag in HASHTAGS:
            if len(seen_links) >= TARGET_TWEETS:
                break

            search_url = f"https://x.com/search?q={quote(tag)}&f=live"
            page.goto(search_url, timeout=60000)
            page.wait_for_selector("article", timeout=100000)
            page.wait_for_timeout(random.randint(2500, 4500))

            empty_scrolls = 0

            while True:
                before_count = len(seen_links)

                # Scroll the page
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.randint(2500, 4000))

                tweets = page.locator("article")
                count = tweets.count()
                logging.info(f"{tag} | Scroll | Articles found in DOM: {count} | Seen links: {before_count}")

                # Parse tweets
                for i in range(min(count, 40)):
                    try:
                        tweet = tweets.nth(i)
                        link_el = tweet.locator("a[href*='/status/']").first
                        if not link_el.count():
                            continue
                        link = link_el.get_attribute("href")
                        if not link or link in seen_links:
                            continue

                        seen_links.add(link)

                        time_el = tweet.locator("time").first
                        if not time_el.count():
                            continue
                        timestamp = time_el.get_attribute("datetime")
                        tweet_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        if tweet_time < CUTOFF_TIME:
                            continue

                        content = normalize_text(tweet.locator("div[data-testid='tweetText']").inner_text())

                        row = {
                            "username": re.search(r"/([^/]+)/status", link).group(1),
                            "timestamp": timestamp,
                            "content": content,
                            "replies": parse_count(tweet.locator('[data-testid="reply"]')),
                            "retweets": parse_count(tweet.locator('[data-testid="retweet"], [data-testid="repost"]')),
                            "likes": parse_count(tweet.locator('[data-testid="like"]')),
                            "mentions": re.findall(r"@\w+", content),
                            "hashtags": [h.lower() for h in re.findall(r"#\w+", content)],
                            "permalink": link,
                            "source_tag": tag
                        }

                        buffer.append(row)

                        if len(buffer) >= CHUNK_SIZE:
                            flush_to_parquet(buffer)
                            buffer.clear()

                        if len(seen_links) >= TARGET_TWEETS:
                            break

                    except Exception:
                        logging.exception("Tweet parse failed")

                # Check if new tweets were added
                after_count = len(seen_links)
                new_tweets = after_count - before_count

                if new_tweets == 0:
                    empty_scrolls += 1
                    logging.info(f"{tag} | No new tweets this scroll ({empty_scrolls}/{REPEAT_THRESHOLD})")
                    if empty_scrolls >= REPEAT_THRESHOLD:
                        logging.info(f"{tag} | No new tweets for {REPEAT_THRESHOLD} scrolls → moving to next hashtag")
                        break
                else:
                    empty_scrolls = 0  # reset counter if new tweets found

    if all_rows:
        flush_to_parquet(all_rows)

    logging.info("✅ Scraping completed successfully.")


if __name__ == "__main__":
    scrape_tweets()
