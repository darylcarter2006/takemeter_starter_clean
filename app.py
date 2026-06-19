"""
TakeMeter — web interface for the fine-tuned NBA discourse classifier.

Prerequisites:
  1. my_model/ must be in this repo root (downloaded from Colab after training)
  2. pip install flask transformers torch

Run:
  python app.py
  Then open http://127.0.0.1:5000
"""

import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ID_TO_LABEL = {0: "analysis", 1: "hot_take", 2: "reaction"}
LABEL_COLORS = {"analysis": "#2563eb", "hot_take": "#dc2626", "reaction": "#16a34a"}

try:
    _tokenizer = AutoTokenizer.from_pretrained("./my_model")
    _model = AutoModelForSequenceClassification.from_pretrained("./my_model")
    _model.eval()
except OSError:
    raise SystemExit(
        "\nModel not found at ./my_model\n"
        "In Colab, run:  trainer.save_model('./my_model')\n"
        "Then download the my_model/ folder and place it in the repo root."
    )

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TakeMeter</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; background: #f9fafb; }
    h1 { font-size: 2rem; margin-bottom: 4px; }
    p.sub { color: #6b7280; margin-top: 0; margin-bottom: 24px; }
    textarea { width: 100%; height: 120px; padding: 12px; font-size: 15px; border: 1px solid #d1d5db; border-radius: 8px; resize: vertical; box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 24px; font-size: 15px; background: #1d4ed8; color: white; border: none; border-radius: 8px; cursor: pointer; }
    button:hover { background: #1e40af; }
    #result { margin-top: 28px; padding: 20px 24px; border-radius: 10px; background: white; border: 1px solid #e5e7eb; display: none; }
    #label { font-size: 1.8rem; font-weight: 700; }
    #conf { color: #6b7280; font-size: 1rem; margin-top: 4px; }
    .examples { margin-top: 32px; }
    .examples h3 { font-size: 1rem; color: #374151; margin-bottom: 10px; }
    .ex { cursor: pointer; padding: 10px 14px; border: 1px solid #e5e7eb; border-radius: 8px; background: white; margin-bottom: 8px; font-size: 14px; color: #374151; }
    .ex:hover { border-color: #93c5fd; background: #eff6ff; }
  </style>
</head>
<body>
  <h1>TakeMeter</h1>
  <p class="sub">Classifies r/nba posts as <b>analysis</b>, <b>hot_take</b>, or <b>reaction</b></p>
  <textarea id="txt" placeholder="Paste an r/nba post or comment here..."></textarea><br>
  <button onclick="classify()">Classify</button>

  <div id="result">
    <div id="label"></div>
    <div id="conf"></div>
  </div>

  <div class="examples">
    <h3>Try an example:</h3>
    <div class="ex" onclick="use(this)">People forget the Warriors' 2022 title was built on their bench outscoring opponents by 8.3 per 100 possessions — compare to 2018 where the bench was -2.1. The title was built on depth, not just the stars.</div>
    <div class="ex" onclick="use(this)">Giannis will never win another ring without a legitimate second star. He's proven he can't carry a team through a 7-game series against real competition.</div>
    <div class="ex" onclick="use(this)">HE DID IT AGAIN. CURRY FROM 40 FEET. I WILL NEVER DOUBT THIS MAN.</div>
    <div class="ex" onclick="use(this)">The Celtics are the most overrated team in the East. They pad their stats against garbage teams all season then fold when it actually matters.</div>
    <div class="ex" onclick="use(this)">Jokic's assist-to-turnover ratio in playoff series he wins vs loses is 4.2 vs 2.8 — the turnovers cluster in late-game possessions when defenses shade toward him.</div>
  </div>

  <script>
    const COLORS = {{ colors|tojson }};
    function use(el) { document.getElementById('txt').value = el.textContent; classify(); }
    async function classify() {
      const text = document.getElementById('txt').value.trim();
      if (!text) return;
      const res = await fetch('/classify', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text}) });
      const data = await res.json();
      const el = document.getElementById('result');
      document.getElementById('label').textContent = data.label;
      document.getElementById('label').style.color = COLORS[data.label] || '#111';
      document.getElementById('conf').textContent = 'Confidence: ' + data.confidence;
      el.style.display = 'block';
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, colors=LABEL_COLORS)


@app.route("/classify", methods=["POST"])
def classify():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"label": "—", "confidence": "—"})
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    label = ID_TO_LABEL[pred_id]
    confidence = f"{probs[pred_id].item():.1%}"
    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False)
