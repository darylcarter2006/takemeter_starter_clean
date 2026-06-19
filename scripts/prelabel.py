"""
Pre-label data/nba_raw.csv using Groq llama-3.3-70b-versatile.

Setup:
  Add GROQ_API_KEY to your .env file (same key as your other projects).
  pip install groq python-dotenv pandas

Input:  data/nba_raw.csv (columns: text, source) from collect_nba.py
Output: data/nba_prelabeled.csv with an added `label_prelabeled` column.

After this script:
  1. Open data/nba_prelabeled.csv in a spreadsheet
  2. Review EVERY row — correct any label you disagree with
  3. Add a `label` column containing your corrected labels
  4. Save the final version as data/nba_labeled.csv (must have: text, label)
  5. Upload data/nba_labeled.csv to the Colab notebook
"""

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

Respond with ONLY the label name: analysis, hot_take, or reaction. Nothing else."""

VALID_LABELS = {"analysis", "hot_take", "reaction"}


def classify(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in VALID_LABELS else "UNPARSEABLE"
    except Exception as e:
        print(f"  API error: {e}")
        return "ERROR"


df = pd.read_csv("data/nba_raw.csv")
print(f"Loaded {len(df)} rows from data/nba_raw.csv")

labels = []
for i, row in df.iterrows():
    label = classify(str(row["text"]))
    labels.append(label)
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(df)} classified...")
    time.sleep(0.15)

df["label_prelabeled"] = labels
df.to_csv("data/nba_prelabeled.csv", index=False)

print(f"\nDistribution of pre-labels:")
print(df["label_prelabeled"].value_counts().to_string())

unparseable = (df["label_prelabeled"] == "UNPARSEABLE").sum()
if unparseable > 0:
    print(f"\nWarning: {unparseable} rows returned UNPARSEABLE — check those manually.")

print(f"\nSaved to data/nba_prelabeled.csv")
print("\nNext steps:")
print("  1. Open data/nba_prelabeled.csv in a spreadsheet")
print("  2. Review every row — correct any label you disagree with")
print("  3. Copy corrected labels into the 'label' column")
print("  4. Save as data/nba_labeled.csv")
print("  5. Check label counts — no single label should exceed 70% of total")
