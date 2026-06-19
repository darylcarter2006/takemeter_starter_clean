# TakeMeter — Planning Document

*Written before data collection, per project spec.*

---

## Community

I chose r/nba for TakeMeter. r/nba is one of the most active sports communities on Reddit, with hundreds of posts and thousands of comments per day ranging from in-game reactions to deep statistical breakdowns. This makes it well-suited for a classification task: the discourse is genuinely varied in structure and intent, and the community has its own established norms around what counts as good discourse.

The distinction between "analysis" and "hot take" is explicitly named and debated within the subreddit — regular users call out posts as hot takes or credit others with doing real analysis. Game threads produce a distinct third type: purely emotional, moment-specific reactions with no argumentative content. These three types emerge naturally from the community itself, not from outside imposition, which means the labels are grounded in how r/nba actually talks about discourse quality.

---

## Labels

I defined three labels that together cover the overwhelming majority of r/nba posts and comments.

### `analysis`

**Definition:** The post makes a structured argument using specific, verifiable statistics, historical comparisons, or tactical observations; the evidence would support the claim even stripped of opinion framing.

**Example 1:**
> "Jokic's assist-to-turnover ratio in playoff series he wins vs. loses is 4.2 vs 2.8 — the turnovers aren't just random variance, they cluster in late-game possessions when he's being pressed. Here's the breakdown by quarter and opponent pressure tier."

**Example 2:**
> "People forget the Warriors' 2022 title was largely built on their bench outscoring opponents by 8.3 per 100 possessions — that's not a superstar stat, that's depth. Compare it to their 2018 run where the bench was -2.1 and you see how much the roster construction changed."

---

### `hot_take`

**Definition:** A bold, confident opinion stated without supporting evidence or with only decorative evidence; the post asserts rather than argues and would not be less persuasive if the specific claim were stated differently.

**Example 1:**
> "Giannis will never win another ring without a legitimate second star. He's proven he can't carry a team through a 7-game series against real competition."

**Example 2:**
> "The Celtics are the most fraudulent dynasty in recent memory. They win regular season games against nobody and fold every playoff run that actually matters."

---

### `reaction`

**Definition:** An immediate emotional response to a specific game, trade, injury, or news event; the post expresses a feeling in the moment with little to no argument, and the specific event is required context to understand the post.

**Example 1:**
> "HE DID IT AGAIN. CURRY FROM 40 FEET. I WILL NEVER DOUBT THIS MAN."

**Example 2:**
> "Can't believe they just traded KAT. What is the front office doing. Season is over before it started."

---

## Hard Edge Cases

### Edge Case 1: The Stat-Backed Hot Take

**Post:** *"LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."*

**Labels it could be:**
- `analysis` — cites a specific, verifiable statistic
- `hot_take` — accusatory framing, the stat is cherry-picked and selected for effect

**Decision rule:** Is the evidence specific and verifiable, and does it *genuinely support* the claim as part of a structured argument — or is it cherry-picked and decorative? A single stat with no surrounding argument, presented to support an accusatory framing, is `hot_take`. The evidence needs to be doing real argumentative work, not just lending credibility to an assertion. This post is `hot_take`.

---

### Edge Case 2: The Event-Triggered Hot Take

**Post:** *"After that loss last night, I'm convinced Luka will never have what it takes to win a ring."*

**Labels it could be:**
- `reaction` — references a specific recent event (the loss)
- `hot_take` — makes a general claim about a player that could stand any week

**Decision rule:** If the post is triggered by a specific event but the claim itself is a general opinion about a player or team, label it `hot_take`. `Reaction` is reserved for posts that are expressions of the moment — where the event is the point, not just the occasion for a broader opinion. A feeling ("I can't believe we lost") is `reaction`; a verdict about a player's career ("he'll never...") triggered by a loss is `hot_take`.

---

### Hard Annotation Examples from Actual Data

*[FILL IN during annotation — add 3 examples that genuinely gave you pause, with which labels they could belong to and what you decided. These must come from real posts in your dataset.]*

**Example 1:**
- Post: [paste text]
- Could be: `X` or `Y`
- Decision: labeled `X` because [reason]

**Example 2:**
- Post: [paste text]
- Could be: `X` or `Y`
- Decision: labeled `X` because [reason]

**Example 3:**
- Post: [paste text]
- Could be: `X` or `Y`
- Decision: labeled `X` because [reason]

---

## Data Collection Plan

**Sources:**
- **Game threads and post-game threads** — primary source of `reaction` comments; these threads are posted during and after every game and generate hundreds of pure emotional responses
- **Hot and new discussion posts** — broad mix, but the opinion posts and trade discussions generate high density of `hot_take` content
- **Keyword searches** ("per 100", "efficiency", "breakdown", "historical", "stats") — targeted `analysis` posts that cite data

**Collection method:** PRAW (Python Reddit API Wrapper) via `scripts/collect_nba.py`. Output is a raw CSV with all collected text, no labels yet.

**Target:** 75 examples per label (225 total), with 300+ rows collected raw to have a buffer for examples that don't fit cleanly or to rebalance if needed.

**If imbalanced after 200+ labeled examples:** If any label accounts for more than 70% of the labeled set, I'll do targeted collection from the source type that generates the underrepresented label — specifically game thread comments for `reaction`, stat-heavy posts for `analysis`, or opinion/trade discussion threads for `hot_take`.

**Minimum balance requirement:** No single label above 70%, each label at least 20% of the final dataset.

---

## Evaluation Metrics

I'll use **per-class F1** as the primary metric, with overall accuracy reported for comparison against the baseline.

**Why not just accuracy:**
For a 3-class task where the label distribution may not be perfectly even, overall accuracy is misleading. A model that predicts the majority class constantly can achieve 50%+ accuracy while failing completely on minority classes. F1 — the harmonic mean of precision and recall per class — catches this failure mode: a class with F1 ≈ 0 means the model learned nothing about that boundary, regardless of overall accuracy.

**Why F1 over precision or recall alone:**
Precision and recall are complementary — high precision with low recall means the model is too conservative; high recall with low precision means it over-predicts. F1 balances both. For a classifier intended to be useful (not just to minimize a single error direction), F1 is the right single number per class.

**Metrics I'll report:**
- Overall accuracy for both models
- Per-class precision, recall, and F1 for both models
- Confusion matrix showing which label pairs the fine-tuned model confuses and in which direction

---

## Definition of Success

The classifier is successful if all three of the following hold:

1. **Fine-tuned model accuracy > 70%** on the test set (random chance for 3 classes = 33%; this threshold means the model is doing meaningful work)
2. **Per-class F1 ≥ 0.65 for all three labels** — no label is being systematically ignored
3. **Fine-tuned model outperforms the Groq zero-shot baseline by at least 10 percentage points** — fine-tuning actually added value beyond a general LLM with a well-written prompt

If the fine-tuned model underperforms the baseline or has a class F1 near zero, that's a signal to investigate annotation consistency or class balance before writing up results — not to declare failure and move on.

---

## AI Tool Plan

### Label Stress-Testing

Before annotating, I'll give an LLM my label definitions and both edge case descriptions and ask it to generate 10–15 posts that sit at the boundary between two labels (specifically `analysis`/`hot_take` and `hot_take`/`reaction` — the two pairs I identified as hard). If it produces posts I can't classify cleanly using my decision rules, my definitions need tightening. This is the right time to find problems.

### Annotation Assistance

I'll use Groq (`llama-3.3-70b-versatile`) to pre-label all collected posts using the classification prompt in `scripts/prelabel.py`. This produces a `label_prelabeled` column. I'll review and correct every row individually before treating any label as ground truth — I won't skim. Pre-labeling without review just introduces the LLM's mistakes into the training data.

I'll track the pre-labeling in the `notes` column (marking rows where I disagreed with Groq) so I can report on it honestly in the AI usage section of the README.

### Failure Analysis

After running the fine-tuned model on the test set, I'll paste all misclassified examples into an LLM and ask it to identify common themes — similar post length, sarcasm, event-triggered framing, ambiguous evidence. I'll verify each suggested pattern by re-reading the examples myself before including it in the evaluation report. Patterns the LLM names but I can't verify by reading the examples will be discarded.

---

## Stretch Features

*Update this section before starting each one.*

### Error Pattern Analysis *(planned)*

After Section 4 in Colab, systematically identify which label pair accounts for the most confusion matrix errors. Go beyond listing individual wrong predictions — find a structural reason (short posts, sarcasm, event-triggered framing, stat-decorated assertions) that explains a cluster of errors.

### Deployed Gradio Interface *(planned)*

After downloading `my_model/` from Colab, run `python app.py` to launch a local interface that accepts any post and returns the predicted label + confidence. Code is at `app.py`; instructions for running it are in the README.
