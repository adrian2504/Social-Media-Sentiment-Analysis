import pickle, pandas as pd, numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

df = pd.read_csv("data/sentimentdataset.csv")
corpus = df["text"].astype(str).tolist()

tok = Tokenizer()
tok.fit_on_texts(corpus)
seqs = []
for line in tok.texts_to_sequences(corpus):
    for i in range(2, len(line)+1):
        seqs.append(line[:i])
max_len = max(len(x) for x in seqs)
seqs = pad_sequences(seqs, maxlen=max_len, padding="pre")
X, y = seqs[:, :-1], seqs[:, -1]
y = to_categorical(y, num_classes=len(tok.word_index)+1)

model = Sequential([
    Embedding(len(tok.word_index)+1, 128, input_length=max_len-1),
    LSTM(256),
    Dense(len(tok.word_index)+1, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=20, batch_size=256)

model.save("models/lstm_nextword.h5")
pickle.dump(tok, open("models/tokenizer.pkl", "wb"))
