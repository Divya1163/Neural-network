"""
Sentiment Analysis Engine.
Provides LSTM training/prediction and notebook-style SimpleRNN artifacts with PKL persistence.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


def clean_text(text):
    """Clean and preprocess text data."""
    text = re.sub(r"http\S+|www.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text


def train_sentiment_model(
    csv_path,
    epochs=20,
    batch_size=16,
    vocab_size=5000,
    embedding_dim=64,
    lstm_units=128,
    dense_units=64,
    max_len=100,
    test_split=0.2,
):
    """Train LSTM sentiment model."""
    try:
        df = pd.read_csv(csv_path, header=None, encoding="latin-1")
        sentiment_col = df[0].values
        text_col = df[5].values

        def map_sentiment(val):
            if val == 4:
                return "positive"
            if val == 0:
                return "negative"
            return "neutral"

        sentiments = [map_sentiment(s) for s in sentiment_col]
        data = pd.DataFrame({"text": text_col, "sentiment": sentiments})

        data = data.dropna(subset=["text"])
        data["text"] = data["text"].astype(str).apply(clean_text)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data["sentiment"].values)

        train_texts, test_texts, y_train, y_test = train_test_split(
            data["text"].values,
            y,
            test_size=test_split,
            random_state=42,
            stratify=y,
        )

        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_texts)

        train_sequences = tokenizer.texts_to_sequences(train_texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)

        x_train = pad_sequences(train_sequences, maxlen=max_len, padding="post")
        x_test = pad_sequences(test_sequences, maxlen=max_len, padding="post")

        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
                LSTM(lstm_units, activation="relu"),
                Dropout(0.3),
                Dense(dense_units, activation="relu"),
                Dropout(0.2),
                Dense(len(label_encoder.classes_), activation="softmax"),
            ]
        )

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        model_info = {
            "label_encoder_classes": label_encoder.classes_.tolist(),
            "max_len": max_len,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
        }

        return {
            "success": True,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "epochs": epochs,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "max_len": max_len,
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "train_loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "history": {
                "accuracy": [float(x) for x in history.history["accuracy"]],
                "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
                "loss": [float(x) for x in history.history["loss"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
            },
            "class_distribution": {
                "positive": int((data["sentiment"] == "positive").sum()),
                "negative": int((data["sentiment"] == "negative").sum()),
                "neutral": int((data["sentiment"] == "neutral").sum()),
                "total": len(data),
            },
            "model_info": model_info,
            "model_object": model,
            "tokenizer": tokenizer,
            "label_encoder": label_encoder,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_sentiment(text, model, tokenizer, label_encoder, max_len=100):
    """Predict with LSTM sentiment model."""
    try:
        cleaned_text = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_len, padding="post")

        predictions = model.predict(padded, verbose=0)
        confidence = float(predictions[0].max())
        predicted_idx = int(predictions[0].argmax())
        predicted_sentiment = label_encoder.classes_[predicted_idx]

        probabilities = {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, predictions[0])
        }

        return {
            "success": True,
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": predicted_sentiment,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _build_notebook_style_model():
    """Build notebook-style SimpleRNN architecture."""
    model = Sequential()
    model.add(Embedding(input_dim=20000, output_dim=2, input_length=35))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_notebook_style_model(
    train_csv_path="rnn_project/train.csv",
    test_csv_path="rnn_project/test.csv",
    fallback_csv_path="rnn_project/testdata.manual.2009.06.14.csv",
    epochs=10,
):
    """Train notebook-style SimpleRNN model."""
    try:
        if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
            train_ds = pd.read_csv(train_csv_path, encoding="latin-1")
            validation_ds = pd.read_csv(test_csv_path, encoding="latin-1")
            train_ds = train_ds.dropna(subset=["text", "sentiment"])[["text", "sentiment"]]
            validation_ds = validation_ds.dropna(subset=["text", "sentiment"])[["text", "sentiment"]]
        else:
            df = pd.read_csv(fallback_csv_path, header=None, encoding="latin-1")
            df = df[[0, 5]].dropna()
            df.columns = ["sentiment", "text"]

            def map_num_to_label(v):
                if int(v) == 4:
                    return "positive"
                if int(v) == 0:
                    return "negative"
                return "neutral"

            df["sentiment"] = df["sentiment"].apply(map_num_to_label)
            train_ds, validation_ds = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["sentiment"],
            )

        train_ds["text"] = train_ds["text"].fillna("").astype(str)
        validation_ds["text"] = validation_ds["text"].fillna("").astype(str)

        def map_sentiment(sentiment):
            sentiment = str(sentiment).strip().lower()
            if sentiment == "positive":
                return 0
            if sentiment == "negative":
                return 1
            return 2

        train_ds["sentiment"] = train_ds["sentiment"].apply(map_sentiment)
        validation_ds["sentiment"] = validation_ds["sentiment"].apply(map_sentiment)

        x_train = np.array(train_ds["text"].tolist())
        y_train = np.array(train_ds["sentiment"].tolist())
        x_test = np.array(validation_ds["text"].tolist())
        y_test = np.array(validation_ds["sentiment"].tolist())

        y_train_cat = to_categorical(y_train, 3)
        y_test_cat = to_categorical(y_test, 3)

        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)
        tokenizer.fit_on_texts(x_test)

        x_train_seq = tokenizer.texts_to_sequences(x_train)
        x_test_seq = tokenizer.texts_to_sequences(x_test)

        max_len = 35
        x_train_pad = pad_sequences(x_train_seq, padding="post", maxlen=max_len)
        x_test_pad = pad_sequences(x_test_seq, padding="post", maxlen=max_len)

        model = _build_notebook_style_model()
        history = model.fit(
            x_train_pad,
            y_train_cat,
            epochs=epochs,
            validation_data=(x_test_pad, y_test_cat),
            verbose=0,
        )

        test_loss, test_acc = model.evaluate(x_test_pad, y_test_cat, verbose=0)

        return {
            "success": True,
            "model_object": model,
            "tokenizer": tokenizer,
            "max_len": max_len,
            "label_map": {0: "positive", 1: "negative", 2: "neutral"},
            "train_samples": int(len(x_train_pad)),
            "test_samples": int(len(x_test_pad)),
            "epochs": int(epochs),
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "history": {
                "accuracy": [float(v) for v in history.history["accuracy"]],
                "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_notebook_style_model_to_pkl(artifacts, pkl_path):
    """Save notebook model artifacts to PKL file."""
    try:
        payload = {
            "version": 1,
            "model_type": "notebook_simple_rnn",
            "weights": artifacts["model_object"].get_weights(),
            "tokenizer": artifacts["tokenizer"],
            "max_len": int(artifacts.get("max_len", 35)),
            "label_map": artifacts.get("label_map", {0: "positive", 1: "negative", 2: "neutral"}),
            "meta": {
                "train_samples": artifacts.get("train_samples"),
                "test_samples": artifacts.get("test_samples"),
                "epochs": artifacts.get("epochs"),
                "train_accuracy": artifacts.get("train_accuracy"),
                "val_accuracy": artifacts.get("val_accuracy"),
                "test_accuracy": artifacts.get("test_accuracy"),
                "test_loss": artifacts.get("test_loss"),
            },
        }

        folder = os.path.dirname(pkl_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {"success": True, "pkl_path": pkl_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def load_notebook_style_model_from_pkl(pkl_path):
    """Load notebook model artifacts from PKL and restore Keras model."""
    try:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        max_len = int(payload.get("max_len", 35))
        model = _build_notebook_style_model()

        dummy = np.zeros((1, max_len), dtype=np.int32)
        model.predict(dummy, verbose=0)
        model.set_weights(payload["weights"])

        return {
            "success": True,
            "model_object": model,
            "tokenizer": payload["tokenizer"],
            "max_len": max_len,
            "label_map": payload.get("label_map", {0: "positive", 1: "negative", 2: "neutral"}),
            "meta": payload.get("meta", {}),
            "pkl_path": pkl_path,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_sentiment_notebook_style(text, model, tokenizer, label_map, max_len=35):
    """Predict with notebook-style SimpleRNN model."""
    try:
        new_text_seq = tokenizer.texts_to_sequences([str(text)])
        new_text_padded = pad_sequences(new_text_seq, padding="post", maxlen=max_len)
        predictions = model.predict(new_text_padded, verbose=0)
        predicted_class_index = int(predictions.argmax(axis=-1)[0])
        sentiment = label_map.get(predicted_class_index, "neutral")

        probabilities = {
            "positive": float(predictions[0][0]),
            "negative": float(predictions[0][1]),
            "neutral": float(predictions[0][2]),
        }

        return {
            "success": True,
            "text": text,
            "cleaned_text": str(text),
            "sentiment": sentiment,
            "confidence": float(np.max(predictions[0])),
            "probabilities": probabilities,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
