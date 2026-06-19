# TakeMeter: NBA Discourse Quality Classifier

**Community:** r/nba | **Model:** DistilBERT (fine-tuned) | **Labels:** `analysis` / `hot_take` / `reaction`

---

## Overview

TakeMeter classifies r/nba posts and comments into three categories that reflect how the community talks about discourse quality. The classifier is fine-tuned from `distilbert-base-uncased` on 200+ manually annotated examples and evaluated against a zero-shot Groq baseline (llama-3.3-70b-versatile).

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

**Source:** r/nba subreddit via the PRAW Python library (`scripts/collect_nba.py`).

Collection strategy:
- Game threads and post-game threads → high density of `reaction` comments
- Hot/new discussion posts → mix of `hot_take` and `analysis` content
- Keyword searches ("per 100", "efficiency", "breakdown", "historical") → targeted `analysis` posts

All examples are public posts and comments. Collected posts were saved to `data/nba_raw.csv`, then pre-labeled with Groq before manual review (see AI usage section).

---

## Labeling Process

**[FILL IN after annotation — replace everything in this section with your actual numbers]**

Label distribution:

| Label | Count | Percentage |
|---|---|---|
| `analysis` | — | —% |
| `hot_take` | — | —% |
| `reaction` | — | —% |
| **Total** | — | 100% |

**Annotation workflow:** Groq (`llama-3.3-70b-versatile`) pre-labeled all collected posts. I reviewed and corrected every pre-assigned label individually before treating it as ground truth.

### Three Difficult-to-Label Examples

**[FILL IN during annotation — add 3 examples that gave you genuine pause]**

**1. [short description]**
- Post: *[paste text here]*
- Could be: `X` or `Y`
- Decision: labeled `X` because [reason]

**2. [short description]**
- Post: *[paste text here]*
- Could be: `X` or `Y`
- Decision: labeled `X` because [reason]

**3. [short description]**
- Post: *[paste text here]*
- Could be: `X` or `Y`
- Decision: labeled `X` because [reason]

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

**Hyperparameter decision — epochs:** I increased epochs from the notebook default of 3 to 5. DistilBERT fine-tuned on ~140 training examples (70% of 200) benefits from more passes through the data — validation loss typically continues dropping through epoch 5 on small datasets, and the risk of overfitting is low given how much general language understanding DistilBERT already carries. **[FILL IN: note whether validation loss actually kept improving or plateaued early]**

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

Results collected via `scripts/prelabel.py` (same client and model, run against the locked test set).

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Groq baseline (zero-shot) | —% |
| DistilBERT fine-tuned | —% |
| Improvement | — pp |

**[FILL IN after Colab — copy numbers from evaluation_results.json]**

### Per-Class Metrics

**Groq Baseline:**

| Label | Precision | Recall | F1 |
|---|---|---|---|
| `analysis` | — | — | — |
| `hot_take` | — | — | — |
| `reaction` | — | — | — |

**DistilBERT Fine-Tuned:**

| Label | Precision | Recall | F1 |
|---|---|---|---|
| `analysis` | — | — | — |
| `hot_take` | — | — | — |
| `reaction` | — | — | — |

**[FILL IN from Colab Section 4 and Section 5 output]**

### Confusion Matrix (Fine-Tuned Model)

Rows = true label, columns = predicted label.

| | Pred: analysis | Pred: hot_take | Pred: reaction |
|---|---|---|---|
| **True: analysis** | — | — | — |
| **True: hot_take** | — | — | — |
| **True: reaction** | — | — | — |

**[FILL IN from confusion_matrix.png / Colab Section 4 output]**

### Wrong Predictions — Analysis

**[FILL IN after Colab — pick 3 misclassified examples from Section 4's wrong-prediction output. For each: paste the post, state the true and predicted labels, and analyze *why* the model got it wrong — not just that it did.]**

**Wrong prediction 1:**
- Post: *[paste text]*
- True label: `X` | Predicted: `Y` | Confidence: —%
- Analysis: [Which labels are being confused? Why is that boundary hard? Is this a labeling inconsistency or a data/boundary problem? What would fix it?]

**Wrong prediction 2:**
- Post: *[paste text]*
- True label: `X` | Predicted: `Y` | Confidence: —%
- Analysis: [...]

**Wrong prediction 3:**
- Post: *[paste text]*
- True label: `X` | Predicted: `Y` | Confidence: —%
- Analysis: [...]

### Sample Classifications

**[FILL IN — run 3–5 posts through the fine-tuned model, report predicted label and confidence as a table. For at least one correct prediction, add a sentence explaining why the prediction is reasonable.]**

| Post (truncated to 80 chars) | Predicted | Confidence |
|---|---|---|
| — | — | —% |
| — | — | —% |
| — | — | —% |

*Correct prediction explained:* [For one entry above, explain in 1–2 sentences why the model's prediction makes sense given the post content and label definition.]

---

## Error Pattern Analysis *(stretch feature)*

**[FILL IN after Colab — identify a systematic pattern in the wrong predictions, not just individual mistakes. Use an LLM to surface candidates, then verify each by reading the examples yourself.]**

*Example structure:* The model consistently confuses `X` with `Y` when [condition]. This is likely because [surface feature the model latched onto] is correlated with `Y` in the training data but doesn't reliably distinguish it from `X`. To fix this, you'd need [more examples / tighter label definition / different training examples that show the hard case explicitly].

---

## Reflection: What the Model Learned vs. What I Intended

**[FILL IN after Colab — write a higher-level observation about the gap between your intended label boundaries and what the model's decision boundary actually captures. This is distinct from the wrong-prediction list above.]**

Prompts to help you write this:
- What surface features (post length, all-caps, question marks, specific player names) might predict the label better than the actual argumentative structure?
- Did the model learn to distinguish argument structure, or did it learn to key on vocabulary (e.g., "per 100", "stats" → analysis; "never", "fraudulent" → hot_take)?
- What would a post need to look like to fool the model — and does that reveal something about what it actually learned?

---

## Spec Reflection

**One way the spec helped:** The spec's canonical "strong taxonomy" example (analysis/hot_take/reaction) was directly usable as my label set, which saved significant design time and meant my labels matched the notebook's built-in Cell 5 exactly. The emphasis on writing decision rules for edge cases before annotating also prevented rework — I caught the "stat-backed hot take" ambiguity early and resolved it with a clear rule before touching any data.

**One way implementation diverged:** The spec suggests manual data collection is often better because it keeps you close to the data. I chose PRAW-based scripted collection instead, because the volume of game thread comments makes manual collection slow and biased toward whatever happens to be visible at collection time. The tradeoff is that I had to be more intentional about source diversity — pulling from game threads, hot posts, and keyword searches separately to avoid the raw collection being dominated by game reactions.

---

## AI Usage

**Instance 1 — Annotation pre-labeling:**
I provided Groq (`llama-3.3-70b-versatile`) with my label definitions and decision rules from `planning.md`, then asked it to classify each collected post with a single label. Groq produced first-pass labels for all ~300 raw posts. I reviewed and corrected every label individually — I did not skim. **[FILL IN: how many did you correct, and what did Groq tend to get wrong?]** Corrected labels are the ones in `data/nba_labeled.csv`.

**Instance 2 — Failure pattern analysis:**
After running the fine-tuned model on the test set, I pasted all misclassified examples into an LLM and asked it to identify common themes. It suggested **[FILL IN: what it suggested]**. I verified this by re-reading the examples myself. **[FILL IN: what you confirmed vs. what you discarded and why.]**

---

## Running the Gradio Interface

**Prerequisite:** Download the fine-tuned model from Colab. After Section 3 (training), run this cell in Colab:

```python
trainer.save_model("./my_model")
```

Then download `my_model/` from the Colab Files panel (right-click → Download as zip). Extract it and place the `my_model/` folder in the repo root.

**Install and run:**

```bash
pip install gradio transformers torch
python app.py
```

Open `http://localhost:7860` in your browser. Paste any r/nba post to see the predicted label and confidence score.
