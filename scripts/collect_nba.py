"""
Collect r/nba posts and comments via pullpush.io (Reddit archive API).
No API key or Reddit credentials needed.

Setup:
  pip install requests pandas

Output: data/nba_raw.csv with columns: text, source
Target: 300+ rows before annotation filtering
"""

import os
import time
import requests
import pandas as pd

BASE = "https://api.pullpush.io/reddit/search"
HEADERS = {"User-Agent": "nba-discourse-research/1.0"}
SLEEP = 1

rows = []
seen = set()


def get_json(url, params=None):
    time.sleep(SLEEP)
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
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


def fetch_posts(params, source):
    """Pull selftext from submissions; return post IDs."""
    data = get_json(f"{BASE}/submission/", params={"subreddit": "nba", **params})
    ids = []
    for post in data.get("data", []):
        selftext = post.get("selftext", "")
        if selftext:
            add(selftext[:800], source)
        if post.get("id"):
            ids.append(post["id"])
    return ids


def fetch_comments_for_post(post_id, source, size=50):
    """Pull top comments from a single post."""
    data = get_json(
        f"{BASE}/comment/",
        params={"link_id": f"t3_{post_id}", "size": size, "sort": "score"},
    )
    for comment in data.get("data", []):
        add(comment.get("body", ""), source)


def fetch_comments_bulk(params, source):
    """Pull comments matching search params directly (no post lookup needed)."""
    data = get_json(f"{BASE}/comment/", params={"subreddit": "nba", **params})
    for comment in data.get("data", []):
        add(comment.get("body", ""), source)


# --- 1. Hot/discussion posts → hot_take and reaction ---
print("Fetching top r/nba posts ...")
hot_ids = fetch_posts({"size": 100, "sort": "score"}, source="hot_discussion")
print(f"  {len(rows)} rows (post selftexts)")

print("Fetching comments from top 5 posts ...")
for post_id in hot_ids[:5]:
    fetch_comments_for_post(post_id, source="hot_discussion", size=60)
    print(f"  {len(rows)} rows ...")

# --- 2. Stats/analysis posts → analysis ---
print("\nFetching stats/efficiency posts ...")
analysis_ids = fetch_posts(
    {"q": "stats efficiency", "size": 100, "sort": "score"},
    source="analysis_search",
)
print(f"  {len(rows)} rows (post selftexts)")

print("Fetching comments from top 3 analysis posts ...")
for post_id in analysis_ids[:3]:
    fetch_comments_for_post(post_id, source="analysis_search", size=40)
    print(f"  {len(rows)} rows ...")

# --- 3. Game threads → reaction comments ---
print("\nFetching game thread posts ...")
game_ids = fetch_posts(
    {"q": "game thread", "size": 10, "sort": "created_utc"},
    source="game_thread",
)
print(f"  Found {len(game_ids)} game threads")

print("Fetching comments from each game thread ...")
for post_id in game_ids:
    fetch_comments_for_post(post_id, source="game_thread", size=40)
    print(f"  {len(rows)} rows ...")

# --- 4. Top r/nba comments directly (catches reaction/hot_take not in posts above) ---
print("\nFetching top r/nba comments directly ...")
fetch_comments_bulk({"size": 100, "sort": "score"}, source="top_comments")
print(f"  {len(rows)} rows total")

# --- Save ---
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("data/nba_raw.csv", index=False)

print(f"\nDone. {len(df)} rows saved to data/nba_raw.csv")
print("\nSource breakdown:")
print(df["source"].value_counts().to_string())
print("\nNext step: python scripts/prelabel.py")
