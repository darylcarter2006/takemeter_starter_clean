# TakeMeter: NBA Discourse Quality Classifier

**Community:** r/nba | **Model:** DistilBERT (fine-tuned) | **Labels:** `analysis` / `hot_take` / `reaction`

---

## Overview

TakeMeter classifies r/nba posts and comments into three categories that reflect how the community talks about discourse quality. The classifier is fine-tuned from `distilbert-base-uncased` on 300 manually annotated examples and evaluated against a zero-shot Groq baseline (llama-3.3-70b-versatile).

---

## Community and Label Taxonomy

**Community:** r/nba on Reddit. The subreddit is active, text-heavy, and has well-established norms around discourse quality — users regularly distinguish "hot takes" from real analysis, and game reactions are a distinct post type with their own conventions. These distinctions are grounded in how r/nba actually talks, not imposed from outside.

### Labels

| Label | Definition |
|---|---|
| `analysis` | Structured argument using specific, verifiable statistics or historical comparisons; evidence supports the claim even stripped of opinion framing |
| `hot_take` | Bold confident opinion without supporting evidence; asserts rather than argues; may cite stats, but only decoratively |
| `reaction` | Immediate emotional response to a specific game, trade, or news event; expresses a feeling in the moment with little to no argument |

**Example posts:**

**analysis:**
> *"People forget the Warriors' 2022 title was largely built on their bench outscoring opponents by 8.3 per 100 possessions — compare it to their 2018 run where the bench was -2.1 and you see how much the roster construction changed."*

**hot_take:**
> *"Giannis will never win another ring without a legitimate second star. He's proven he can't carry a team through a 7-game series against real competition."*

**reaction:**
> *"HE DID IT AGAIN. CURRY FROM 40 FEET. I WILL NEVER DOUBT THIS MAN."*

---

## Data Collection

**Source:** r/nba subreddit via the pullpush.io Reddit archive API (`scripts/collect_nba.py`). The public Reddit JSON endpoints returned 403 errors; pullpush.io provides access to the same data with no authentication required.

Collection strategy:
- Top-scored r/nba posts and their comments → mix of `hot_take` and `analysis` content
- Keyword searches ("stats efficiency") targeting `analysis` posts with their top comment threads
- Game thread posts and their top comments → high density of `reaction` content
- Direct top-scored r/nba comments → catches `reaction` and `hot_take` not in the post threads above

529 rows were collected before deduplication and filtering; 300 were retained for annotation.

---

## Labeling Process

Label distribution after annotation and 8 manual corrections:

| Label | Count | Percentage |
|---|---|---|
| `analysis` | 157 | 52.3% |
| `hot_take` | 84 | 28.0% |
| `reaction` | 59 | 19.7% |
| **Total** | **300** | 100% |

**Annotation workflow:** `llama-3.1-8b-instant` (Groq) pre-labeled all 300 posts using the same label definitions from `planning.md`, with confidence self-reporting. I reviewed every pre-assigned label and made 8 corrections — the model consistently confused stat-linked reactions with analysis, and event-triggered opinions with reactions. Corrected labels are the ground truth in `data/nba_labeled.csv`.

### Three Difficult-to-Label Examples

**1. Stat-linked reaction that reads like analysis**
- Post: *"It was a 14 point game at half, and Jokic completely disappeared https://www.statmuse.com/nba/ask/jokic-stats-in-the-second-half-of-game-7-vs-thunder"*
- Could be: `reaction` or `analysis`
- Decision: labeled `reaction` because this is an immediate in-the-moment observation from a specific game 7. The stat link supports the emotional claim; it's not building a structured argument. A real analysis post would use the data to argue a broader point across games.

**2. Event-triggered claim with superlative framing**
- Post: *"After losing game 3, SGA proceeded to laugh as if he knew he would end the Nuggets life in game 7. Was that one of the most coldest moments in NBA history and did it show SGA's mental strength?"*
- Could be: `reaction` or `hot_take`
- Decision: labeled `hot_take` because the core claim — "coldest moment in NBA history" — is a general superlative verdict about SGA's legacy, not just an expression of the moment. The game 3 incident is an occasion for a broader opinion about SGA, not the point itself.

**3. Post-game question with structured framing**
- Post: *"Just finished watching OKC/DEN and I am left with one question, is hand checking legal now? I was not rooting for either team going in, was just stoked for this game because the series has been awesome..."*
- Could be: `reaction` or `analysis`
- Decision: labeled `reaction` because this is immediate post-game exasperation, not systematic analysis of officiating rules. The question form is frustration framing; there's no structured argument.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (66M parameters, pre-trained on BooksCorpus + English Wikipedia)

**Training setup:**

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 256 tokens |
| Train / val / test split | 70% / 15% / 15% (stratified) |

**Hyperparameter decision — epochs:** I increased epochs from the notebook default of 3 to 5. The training log confirms this was the right call — validation accuracy kept improving through epoch 5:

| Epoch | Val Accuracy | Val Loss |
|---|---|---|
| 1 | 57.8% | 0.933 |
| 2 | 57.8% | 0.845 |
| 3 | 68.9% | 0.769 |
| 4 | 71.1% | 0.723 |
| 5 | **73.3%** | 0.678 |

The model had not plateaued at epoch 3; it was still learning fast. Stopping early would have cost roughly 5 percentage points of validation accuracy.

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile`, zero-shot (no task-specific training)

**Classification prompt:**

```
You are classifying r/nba posts for a discourse quality study.

Assign exactly one label:
- analysis: structured argument using specific, verifiable statistics or historical
  comparisons; evidence would support the claim without the opinion framing
- hot_take: bold confident opinion without supporting evidence; asserts rather than
  argues; may cite stats only decoratively
- reaction: immediate emotional response to a specific game, trade, or news event;
  expresses a feeling in the moment

Rules:
- A post with one cherry-picked stat and accusatory framing → hot_take
- A post anchored to a specific recent event with no argument → reaction
- A post building a case with multiple specific, verifiable data points → analysis

Respond with ONLY the label name: analysis, hot_take, or reaction. Nothing else.
```

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Groq baseline (zero-shot) | 71.1% |
| DistilBERT fine-tuned | 66.7% |
| Improvement | −4.4 pp |

The fine-tuned model underperformed the Groq baseline by 4.4 percentage points. This is an honest result discussed in the reflection section below.

### Per-Class Metrics

**DistilBERT Fine-Tuned** (derived from confusion matrix):

| Label | Precision | Recall | F1 |
|---|---|---|---|
| `analysis` | 0.73 | 0.92 | 0.81 |
| `hot_take` | 0.50 | 0.58 | 0.54 |
| `reaction` | 1.00 | 0.11 | 0.20 |

**Groq Baseline:**

| Label | Precision | Recall | F1 |
|---|---|---|---|
| `analysis` | 0.86 | 0.79 | 0.83 |
| `hot_take` | 0.60 | 0.50 | 0.55 |
| `reaction` | 0.54 | 0.78 | 0.64 |

### Confusion Matrix (Fine-Tuned Model)

Rows = true label, columns = predicted label.

| | Pred: analysis | Pred: hot_take | Pred: reaction |
|---|---|---|---|
| **True: analysis** | 22 | 2 | 0 |
| **True: hot_take** | 5 | 7 | 0 |
| **True: reaction** | 3 | 5 | 1 |

The model almost never predicts `reaction` — it classified 8 of 9 true reactions as either `analysis` (3) or `hot_take` (5). Only one reaction post was correctly identified.

### Wrong Predictions — Analysis

**Wrong prediction 1:**
- Post: *"It was a 14 point game at half, and Jokic completely disappeared"*
- True label: `reaction` | Predicted: `hot_take` | Confidence: 40.1%
- Analysis: This is an immediate game-7 observation with no argument — a clear reaction. The word "disappeared" is strong emotional language that the model associates with opinionated `hot_take` posts. The model doesn't have signal about event-anchoring; it just sees assertive language and predicts `hot_take`. The low confidence (40.1%) shows the model was genuinely uncertain — it was essentially guessing between the two.

**Wrong prediction 2:**
- Post: *"HE DID IT AGAIN. CURRY FROM 40 FEET. I WILL NEVER DOUBT THIS MAN."*
- True label: `reaction` | Predicted: `hot_take` | Confidence: 45.0%
- Analysis: All-caps excitement is a surface feature shared by both reactions ("I can't believe that shot!") and hot takes ("LEBRON WILL NEVER BE THE GOAT"). The model learned to key on emphatic, exclamatory language as a `hot_take` signal — but reactions are also emphatic. The distinction (is this tied to a specific moment?) requires external context the model doesn't have.

**Wrong prediction 3:**
- Post: *"T-Wolves: 237,156,897 (1st in league), OKC: 165,034,408 (26th in league), Pacers: 171,032,577 (21st in league). OKC can hold on to this core for a while, and the cap space will be massive."*
- True label: `hot_take` | Predicted: `analysis` | Confidence: 46.3%
- Analysis: This post lists specific numbers (salary figures) to assert a conclusion about OKC's future. The model sees numbers and structured formatting → predicts `analysis`. But this is a hot take: the payroll numbers are decorative evidence for a confident assertion. The difference (is the data *doing argumentative work* or just providing decoration for a conclusion?) requires understanding argument structure, not pattern-matching on numeric formatting.

### Sample Classifications

| Post (first 70 chars) | True Label | Predicted | Confidence |
|---|---|---|---|
| "People forget the Warriors' 2022 title was built on their benc..." | `analysis` | `analysis` ✓ | 75.2% |
| "Giannis will never win another ring without a legitimate second..." | `hot_take` | `hot_take` ✓ | 46.9% |
| "HE DID IT AGAIN. CURRY FROM 40 FEET. I WILL NEVER DOUBT THIS..." | `reaction` | `hot_take` ✗ | 45.0% |
| "It was a 14 point game at half, and Jokic completely disappeared" | `reaction` | `hot_take` ✗ | 40.1% |
| "T-Wolves: 237,156,897 (1st in league), OKC: 165,034,408 (26th..." | `hot_take` | `analysis` ✗ | 46.3% |

*Correct prediction explained:* The Warriors bench comparison (post 1) was correctly predicted as `analysis` because it includes specific verifiable stats (8.3 per 100 possessions, -2.1) that directly support a comparative claim — exactly what the model learned to associate with `analysis`. The 75.2% confidence is the model's highest in this table, which makes sense for a clear-cut case.

---

## Error Pattern Analysis *(stretch feature)*

The confusion matrix reveals a clear systematic pattern: **the model collapsed `reaction` into `hot_take`.**

- `reaction` recall: 11% (1 correct out of 9)
- `reaction` precision: 100% (when it does predict reaction, it's right — but it almost never does)
- 5 of 9 true reactions were classified as `hot_take`; 3 were classified as `analysis`

The pattern: any post with emphatic, assertive language and no statistics gets labeled `hot_take`. Reaction posts are often emphatic and assertive (they're emotional responses), so the model can't distinguish them from hot takes without knowing whether the post is anchored to a specific recent event.

The same pattern explains the `hot_take` → `analysis` confusion (5 cases): posts that use numbers or team comparisons as decoration for an opinion get labeled `analysis` because the model learned that numbers → analysis.

**Root cause:** The model learned surface vocabulary features rather than discourse structure. It can detect "has specific stats + comparative framing" → `analysis`, and "assertive + no stats" → `hot_take`. But the `reaction` vs. `hot_take` distinction requires knowing whether the post is anchored to a specific moment in time — which is a contextual/temporal feature that doesn't appear in the post text itself.

**Fix:** Adding timestamps or event tags to training examples, or adding reaction examples that more clearly signal event-anchoring (e.g., "just now", "this game", "right now") would help. Alternatively, collecting more reaction examples (only 19.7% of the dataset) so the model sees more signal for this class.

---

## Reflection: What the Model Learned vs. What I Intended

I designed the labels around **discourse structure** — does this post make an argument? Is it event-anchored? Does it assert without evidence? The model learned **vocabulary features** instead.

Evidence for this:
- `analysis`: model keys on number words, "per 100", "efficiency", comparative language like "vs." and "compared to" → gets 92% recall but 73% precision (over-applies the label)
- `hot_take`: model keys on "will never", "fraudulent", "can't", "worst" → 58% recall, 50% precision
- `reaction`: model has essentially no reliable signal → 11% recall, 100% precision (only predicts it for the most obvious cases)

What would fool the model:
- A calm, structured post saying "I just watched game 7 and wanted to break down why the officiating in Q4 was systematically inconsistent: [list of calls]" — this is actually a `reaction` (immediate post-game), but the model would label it `analysis` because of the structured framing
- A hot take that happens to include a salary table — the model sees numbers and calls it `analysis`

The fundamental problem: r/nba reaction posts and hot takes use similar emotional vocabulary. The feature that separates them ("did this happen 30 minutes ago?") requires context outside the post text. DistilBERT has no way to know whether "Jokic disappeared" was written during game 7 or three years later.

The Groq baseline (71.1%) outperformed the fine-tuned model (66.7%) because a large language model can reason about the argumentative structure of a post more directly — it doesn't have to learn from examples that it's picking up noisy surface cues from. Fine-tuning on 300 examples was not enough to teach structural distinctions that required understanding the post's relationship to a specific moment in time.

---

## Spec Reflection

**One way the spec helped:** The spec's canonical "strong taxonomy" example (analysis/hot_take/reaction) was directly usable as my label set, which saved significant design time and meant my labels matched the notebook's Cell 5 exactly. The emphasis on writing decision rules for edge cases before annotating also prevented rework — I caught the "stat-backed hot take" ambiguity early and resolved it with a clear rule before touching any data.

**One way implementation diverged:** The spec suggests manual data collection is often better because it keeps you close to the data. I chose scripted collection via the pullpush.io archive API instead, because game thread comments make manual collection slow and biased toward whatever is visible at the time. The tradeoff is that I had to be more intentional about source diversity — pulling from game threads, hot posts, and keyword searches separately to avoid the raw collection being dominated by one label.

---

## AI Usage

**Instance 1 — Annotation pre-labeling:**
I provided Groq (`llama-3.1-8b-instant`) with the label definitions and decision rules from `planning.md`, then asked it to classify each of the 300 collected posts with a single label and a confidence level. I reviewed every label individually. I made 8 corrections — Groq most often confused stat-linked reactions with analysis (it saw a stat link and called it analysis even when the post was clearly an immediate game observation). Corrected labels are the ground truth in `data/nba_labeled.csv`.

**Instance 2 — Failure pattern analysis:**
After running the fine-tuned model on the test set, I examined the confusion matrix and ran the model against representative posts to identify systematic patterns. The analysis pointed to the `reaction` → `hot_take` collapse as the dominant failure mode. I verified this by reading the actual wrong-prediction examples and confirming that emphatic emotional language without event-anchoring context was the shared surface feature causing the confusion.

---

## Running the Gradio Interface

**Prerequisite:** The fine-tuned model must be in `my_model/` at the repo root. After training in Colab:

```python
trainer.save_model("./my_model")
```

Download `my_model/` from the Colab Files panel and place it in the repo root. (The `my_model/` directory is excluded from git via `.gitignore` due to size.)

**Install and run:**

```bash
pip install gradio transformers torch
python app.py
```

Open `http://localhost:7860` in your browser. Paste any r/nba post to see the predicted label and confidence score.
