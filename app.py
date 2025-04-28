
import os
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"

import json, pickle
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import keras                                
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── LSTM next-word model
lstm = keras.saving.load_model("models/lstm_nextword.keras")
tok = pickle.load(open("models/tokenizer.pkl", "rb"))
max_len = lstm.input_shape[1]

def next_words(seed: str, n: int = 10) -> str:
    """Generate `n` new words after the `seed` phrase."""
    for _ in range(n):
        seq = pad_sequences(
            [tok.texts_to_sequences([seed])[0]], maxlen=max_len - 1
        )
        wid = lstm.predict(seq, verbose=0).argmax()
        seed += " " + tok.index_word.get(wid, "")
    return seed

# ── BERT sentiment pipeline (PyTorch) 
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    framework="pt"                    # force PyTorch backend
)

def predict_sentiment(text: str):
    out = sentiment_pipe(text, truncation=True)[0]
    raw_label, score = out["label"], out["score"]

    # map RoBERTa’s LABEL_0/1/2 → human-readable
    pretty = {"LABEL_0": "negative",
              "LABEL_1": "neutral",
              "LABEL_2": "positive"}.get(raw_label, raw_label)

    mood = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(pretty, "🤔")
    return pretty, score, mood

# ── Flask setup ──────────────────────────────────────────────────
# ── Flask setup ──────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def root():                         #  ← default = Overview page
    return render_template("overview.html")

@app.route("/overview")
def overview():
    return render_template("overview.html")

@app.route("/charts")
def charts():
    return render_template("charts.html")

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")   # old Live-Mood UI

# ── REST API endpoints ────────────────────────────────────────────
@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    """
    JSON-in  : { "text": "<user post>" }
    JSON-out : { "label": "positive|neutral|negative",
                 "score": 0.97,
                 "emoji": "😊" }
    """
    data = request.get_json(silent=True) or {}
    text  = data.get("text", "")
    label, score, mood = predict_sentiment(text)
    return jsonify({"label": label, "score": score, "emoji": mood})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    JSON-in  : { "seed": "hello world" }
    JSON-out : { "generated": "hello world …" }
    """
    data = request.get_json(silent=True) or {}
    seed = data.get("seed", "")
    return jsonify({"generated": next_words(seed)})



# ── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)