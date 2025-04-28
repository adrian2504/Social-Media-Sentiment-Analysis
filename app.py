
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
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    txt = request.json.get("text", "")
    label, score, mood = predict_sentiment(txt)
    return jsonify({"label": label, "score": score, "emoji": mood})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    seed = request.json.get("seed", "")
    return jsonify({"generated": next_words(seed)})

# ── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)