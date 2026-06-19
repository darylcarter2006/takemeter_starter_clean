"""
TakeMeter — Gradio interface for the fine-tuned NBA discourse classifier.

Prerequisites:
  1. Download my_model/ from Colab after training:
       In Colab, run: trainer.save_model("./my_model")
       Then Files panel → right-click my_model/ → Download as zip
       Extract and place my_model/ in this repo root.
  2. pip install gradio transformers torch

Run:
  python app.py
  Then open http://localhost:7860
"""

import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ID_TO_LABEL = {0: "analysis", 1: "hot_take", 2: "reaction"}

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


def classify_post(text: str):
    if not text.strip():
        return "—", "—"
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    label = ID_TO_LABEL[pred_id]
    confidence = f"{probs[pred_id].item():.1%}"
    return label, confidence


demo = gr.Interface(
    fn=classify_post,
    inputs=gr.Textbox(
        label="r/nba Post or Comment",
        placeholder="Paste an r/nba post or comment here...",
        lines=4,
    ),
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.Textbox(label="Confidence"),
    ],
    title="TakeMeter",
    description="Classifies r/nba posts as **analysis**, **hot_take**, or **reaction**.",
    examples=[
        [
            "Jokic's assist-to-turnover ratio in playoff series he wins vs. loses is 4.2 vs 2.8 "
            "— the turnovers cluster in late-game possessions when he's being pressed."
        ],
        [
            "Giannis will never win another ring without a legitimate second star. "
            "He's proven he can't carry a team through a 7-game series against real competition."
        ],
        ["HE DID IT AGAIN. CURRY FROM 40 FEET. I WILL NEVER DOUBT THIS MAN."],
    ],
)

if __name__ == "__main__":
    demo.queue().launch()
