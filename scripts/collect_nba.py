"""
Collect r/nba posts and comments using Reddit's public JSON API.
No API key or PRAW needed.

Setup:
  pip install requests pandas

Endpoints:
  hot.json                             → hot_take and reaction content
  search.json?q=stats+efficiency       → analysis content
  search.json?q=game+thread            → game thread IDs (reaction comments)
  comments/{id}.json                   → top-level comments per post

Output: data/nba_raw.csv with columns: text, source
Target: 300+ rows before annotation filtering
"""

import os
import time
import requests
import pandas as pd

BASE = "https://www.reddit.com"
SLEEP = 2

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})

# Warm up: load the HTML page so Reddit issues its anonymous session cookie.
# Without this cookie the JSON endpoints return 403.
print("Establishing session with Reddit ...")
session.headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
session.get(f"{BASE}/r/nba/", timeout=15)
session.headers["Accept"] = "application/json, text/javascript, */*"
time.sleep(SLEEP)

rows = []
seen = set()


def get_json(url, params=None):
    time.sleep(SLEEP)
    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def add(text, source):
    text = text.strip()
    if text in ("[removed]", "[deleted]", ""):
        return
    if len(text) < 30 or len(text) > 800:
        return
    if text in seen:
        return
    seen.add(text)
    rows.append({"text": text, "source": source})


def fetch_listing(url, params, source):
    """Pull selftext from text posts in a listing; return all post IDs."""
    data = get_json(url, params)
    post_ids = []
    for child in data["data"]["children"]:
        post = child.get("data", {})
        if post.get("is_self") and post.get("selftext"):
            add(post["selftext"][:800], source)
        if post.get("id"):
            post_ids.append(post["id"])
    return post_ids


def fetch_comments(post_id, source, limit=50):
    """Pull top-level comments from a single post."""
    url = f"{BASE}/r/nba/comments/{post_id}.json"
    data = get_json(url, params={"limit": limit, "sort": "top", "depth": 1})
    if not isinstance(data, list) or len(data) < 2:
        return
    for child in data[1]["data"]["children"]:
        body = child.get("data", {}).get("body", "")
        add(body, source)


# --- 1. Hot posts: hot_take and reaction ---
print("Fetching /r/nba/hot.json ...")
hot_ids = fetch_listing(
    f"{BASE}/r/nba/hot.json",
    params={"limit": 100},
    source="hot_discussion",
)
print(f"  {len(rows)} rows so far (from hot post selftexts)")

print("Fetching comments from top 5 hot posts ...")
for post_id in hot_ids[:5]:
    fetch_comments(post_id, source="hot_discussion")
    print(f"  {len(rows)} rows so far ...")

# --- 2. Analysis keyword search ---
print("\nFetching stats/efficiency search results ...")
analysis_ids = fetch_listing(
    f"{BASE}/r/nba/search.json",
    params={
        "q": "stats efficiency",
        "sort": "top",
        "t": "year",
        "limit": 100,
        "restrict_sr": "on",
    },
    source="analysis_search",
)
print(f"  {len(rows)} rows so far (from analysis post selftexts)")

print("Fetching comments from top 3 analysis posts ...")
for post_id in analysis_ids[:3]:
    fetch_comments(post_id, source="analysis_search")
    print(f"  {len(rows)} rows so far ...")

# --- 3. Game threads: reaction comments ---
print("\nSearching for game threads ...")
game_data = get_json(
    f"{BASE}/r/nba/search.json",
    params={
        "q": "game thread",
        "sort": "new",
        "limit": 10,
        "restrict_sr": "on",
    },
)
game_ids = [
    c["data"]["id"]
    for c in game_data["data"]["children"]
    if c.get("data", {}).get("id")
]
print(f"  Found {len(game_ids)} game threads")

print("Fetching comments from each game thread ...")
for post_id in game_ids:
    fetch_comments(post_id, source="game_thread", limit=40)
    print(f"  {len(rows)} rows so far ...")

# --- Save ---
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("data/nba_raw.csv", index=False)

print(f"\nDone. {len(df)} rows saved to data/nba_raw.csv")
print("\nSource breakdown:")
print(df["source"].value_counts().to_string())
print("\nNext step: python scripts/prelabel.py")
