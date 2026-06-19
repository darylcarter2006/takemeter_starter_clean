"""
Pre-label data/nba_raw.csv using Groq and export a review-ready nba_labeled.csv.

Setup:
  GROQ_API_KEY in .env, pip install groq python-dotenv pandas

Output: data/nba_labeled.csv — label column already filled from Groq.
  Only fix rows where needs_review == True before uploading to Colab.
"""

import json
import os
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are classifying r/nba posts for a discourse quality study.

Assign exactly one label:
- analysis: structured argument using specific, verifiable statistics or historical comparisons; evidence would support the claim without the opinion framing
- hot_take: bold confident opinion without supporting evidence; asserts rather than argues; may cite stats but only decoratively
- reaction: immediate emotional response to a specific game, trade, or news event; expresses a feeling in the moment

Rules:
- A post with one cherry-picked stat and accusatory framing → hot_take
- A post anchored to a specific recent event with no argument → reaction
- A post building a case with multiple specific, verifiable data points → analysis

Respond with JSON only: {"label": "<label>", "confidence": "<high|medium|low>"}
Use confidence=low when the post is ambiguous between two labels."""

VALID_LABELS = {"analysis", "hot_take", "reaction"}
CONFIDENCE_MAP = {"high": 0.95, "medium": 0.65, "low": 0.30}
CONFIDENCE_THRESHOLD = 0.80
SHORT_TEXT_CHARS = 20


def classify(text):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(text)},
            ],
            max_tokens=30,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        label = data.get("label", "").strip().lower()
        conf_str = data.get("confidence", "").strip().lower()
        confidence = CONFIDENCE_MAP.get(conf_str)
        return (label if label in VALID_LABELS else "UNPARSEABLE"), confidence
    except Exception as e:
        print(f"  API error: {e}")
        return "ERROR", None


df = pd.read_csv("data/nba_raw.csv")
print(f"Loaded {len(df)} rows\n")

labels, confidences = [], []
for i, row in df.iterrows():
    label, confidence = classify(row["text"])
    labels.append(label if label in VALID_LABELS else "UNPARSEABLE")
    confidences.append(confidence)
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(df)} classified...")
    time.sleep(0.15)

df["label"] = labels
df["confidence"] = confidences
df["needs_review"] = (
    ~df["label"].isin(VALID_LABELS)
    | df["confidence"].isna()
    | (df["confidence"] < CONFIDENCE_THRESHOLD)
    | (df["text"].str.len() < SHORT_TEXT_CHARS)
)

# --- Distribution ---
print("\nLabel distribution:")
total = len(df)
for label, count in df["label"].value_counts().items():
    pct = count / total * 100
    warning = "  ⚠ OVER 70% — collect more of the other labels" if pct > 70 else ""
    print(f"  {label:12s}  {count:4d}  ({pct:.1f}%){warning}")

# --- Flagged rows ---
flagged = df[df["needs_review"]]
print(f"\n{len(flagged)} of {total} rows flagged for review:")
if not flagged.empty:
    print(f"\n  {'Row':>4}  {'Label':12}  {'Conf':6}  Text preview")
    print(f"  {'-'*4}  {'-'*12}  {'-'*6}  {'-'*50}")
    for idx, row in flagged.iterrows():
        conf = f"{row['confidence']:.0%}" if row["confidence"] is not None else "N/A "
        preview = str(row["text"])[:55].replace("\n", " ")
        print(f"  {idx:>4}  {row['label']:12}  {conf:6}  {preview}")

# --- Save ---
os.makedirs("data", exist_ok=True)
df[["text", "label", "source", "confidence", "needs_review"]].to_csv(
    "data/nba_labeled.csv", index=False
)

print(f"\nSaved data/nba_labeled.csv")
if not flagged.empty:
    print(f"Open it, fix the {len(flagged)} flagged rows, then upload to Colab.")
else:
    print("No rows flagged — upload data/nba_labeled.csv to Colab directly.")
