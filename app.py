import json, pickle, pandas as pd
from flask import Flask, render_template, request, jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from transformers import pipeline

# ----- Models ----------------------------------------------------
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True
)

tok = pickle.load(open("models/tokenizer.pkl", "rb"))
lstm = load_model("models/lstm_nextword.h5")
max_len = lstm.input_shape[1]

# ----- Inference helpers ----------------------------------------
def predict_sentiment(text: str):
    out = sentiment_pipe(text)[0]
    label, score = out["label"], out["score"]
    mood = {
        "positive": "😊",
        "neutral":  "😐",
        "negative": "😞"
    }.get(label.split()[0].lower(), "🤔")
    return label, score, mood

def next_words(seed: str, n: int = 10):
    for _ in range(n):
        seq = pad_sequences([tok.texts_to_sequences([seed])[0]], maxlen=max_len-1)
        wid = lstm.predict(seq, verbose=0).argmax()
        seed += " " + tok.index_word.get(wid, "")
    return seed

# ----- Flask -----------------------------------------------------
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

if __name__ == "__main__":
    app.run(debug=True)
