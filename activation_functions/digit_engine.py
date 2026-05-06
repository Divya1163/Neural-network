"""Single-layer MNIST project backend for dashboard interactions."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


_MNIST_CACHE: tuple[np.ndarray, np.ndarray] | None = None
_LAST_MODELS: dict[str, Any] = {}


def project_config() -> dict[str, Any]:
    return {
        "binary_activations": ["binary_step", "sigmoid", "tanh", "relu", "leaky_relu"],
        "default": {
            "binary_activation": "sigmoid",
            "target_digit": 0,
            "train_limit": 9000,
            "test_limit": 2000,
            "binary_epochs": 35,
            "multiclass_epochs": 25,
            "learning_rate": 0.08,
            "alpha_leaky": 0.01,
        },
    }


def train_project(
    binary_activation: str,
    target_digit: int,
    train_limit: int,
    test_limit: int,
    binary_epochs: int,
    multiclass_epochs: int,
    learning_rate: float,
    alpha_leaky: float,
    random_state: int = 42,
) -> dict[str, Any]:
    if binary_activation not in project_config()["binary_activations"]:
        raise ValueError("Unsupported binary activation selected.")
    if target_digit < 0 or target_digit > 9:
        raise ValueError("target_digit must be in [0, 9].")

    X_full, y_full = _load_mnist()

    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_full,
    )

    train_limit = int(max(1000, min(train_limit, X_train.shape[0])))
    test_limit = int(max(500, min(test_limit, X_test.shape[0])))

    X_train = X_train[:train_limit]
    y_train_int = y_train_int[:train_limit]
    X_test = X_test[:test_limit]
    y_test_int = y_test_int[:test_limit]

    # --- Binary model for selected target digit ---
    y_train_bin = (y_train_int == target_digit).astype(np.float32).reshape(-1, 1)
    y_test_bin = (y_test_int == target_digit).astype(np.float32).reshape(-1, 1)

    Wb = np.zeros((X_train.shape[1], 1), dtype=np.float32)
    bb = 0.0

    binary_train_loss = []
    binary_test_loss = []
    binary_train_acc = []
    binary_test_acc = []

    for _ in range(int(binary_epochs)):
        z = X_train @ Wb + bb

        if binary_activation == "binary_step":
            a = (z >= 0).astype(np.float32)
            err = y_train_bin - a
            dW = -(X_train.T @ err) / X_train.shape[0]
            db = -np.mean(err)
            Wb -= float(learning_rate) * dW
            bb -= float(learning_rate) * float(db)
        else:
            a, da_dz = _binary_activation_and_grad(binary_activation, z, float(alpha_leaky))
            diff = a - y_train_bin
            dz = (2.0 / X_train.shape[0]) * diff * da_dz
            dW = X_train.T @ dz
            db = np.sum(dz)
            Wb -= float(learning_rate) * dW
            bb -= float(learning_rate) * float(db)

        train_eval = _binary_activation_forward(binary_activation, X_train @ Wb + bb, float(alpha_leaky))
        test_eval = _binary_activation_forward(binary_activation, X_test @ Wb + bb, float(alpha_leaky))

        train_loss = float(np.mean((train_eval - y_train_bin) ** 2))
        test_loss = float(np.mean((test_eval - y_test_bin) ** 2))

        if binary_activation == "tanh":
            train_pred = (train_eval >= 0.0).astype(np.float32)
            test_pred = (test_eval >= 0.0).astype(np.float32)
        else:
            train_pred = (train_eval >= 0.5).astype(np.float32)
            test_pred = (test_eval >= 0.5).astype(np.float32)

        train_acc = float(np.mean(train_pred == y_train_bin))
        test_acc = float(np.mean(test_pred == y_test_bin))

        binary_train_loss.append(train_loss)
        binary_test_loss.append(test_loss)
        binary_train_acc.append(train_acc)
        binary_test_acc.append(test_acc)

    # --- Multiclass softmax single-layer model ---
    y_train_oh = np.eye(10)[y_train_int]
    y_test_oh = np.eye(10)[y_test_int]

    Wm = np.zeros((X_train.shape[1], 10), dtype=np.float32)
    bm = np.zeros((1, 10), dtype=np.float32)

    multi_train_loss = []
    multi_test_loss = []
    multi_train_acc = []
    multi_test_acc = []

    for _ in range(int(multiclass_epochs)):
        logits = X_train @ Wm + bm
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        eps = 1e-9
        loss = -np.mean(np.sum(y_train_oh * np.log(probs + eps), axis=1))

        dlogits = (probs - y_train_oh) / X_train.shape[0]
        dW = X_train.T @ dlogits
        db = np.sum(dlogits, axis=0, keepdims=True)

        Wm -= float(learning_rate) * dW
        bm -= float(learning_rate) * db

        train_probs = _softmax(X_train @ Wm + bm)
        test_probs = _softmax(X_test @ Wm + bm)

        train_loss = -np.mean(np.sum(y_train_oh * np.log(train_probs + eps), axis=1))
        test_loss = -np.mean(np.sum(y_test_oh * np.log(test_probs + eps), axis=1))

        train_pred = np.argmax(train_probs, axis=1)
        test_pred = np.argmax(test_probs, axis=1)

        train_acc = float(np.mean(train_pred == y_train_int))
        test_acc = float(np.mean(test_pred == y_test_int))

        multi_train_loss.append(float(train_loss if np.isfinite(train_loss) else loss))
        multi_test_loss.append(float(test_loss))
        multi_train_acc.append(train_acc)
        multi_test_acc.append(test_acc)

    test_pred_multi = np.argmax(_softmax(X_test @ Wm + bm), axis=1)
    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test_int, test_pred_multi):
        confusion[int(t), int(p)] += 1

    # --- One-vs-rest binary accuracy for each class ---
    per_digit_binary_accuracy: list[float] = []
    per_digit_outputs: list[float] = []

    for digit in range(10):
        y_tr_d = (y_train_int == digit).astype(np.float32).reshape(-1, 1)
        y_te_d = (y_test_int == digit).astype(np.float32).reshape(-1, 1)

        Wd = np.zeros((X_train.shape[1], 1), dtype=np.float32)
        bd = 0.0

        for _ in range(max(6, int(binary_epochs // 2))):
            zd = X_train @ Wd + bd
            if binary_activation == "binary_step":
                ad = (zd >= 0).astype(np.float32)
                err = y_tr_d - ad
                dW = -(X_train.T @ err) / X_train.shape[0]
                db = -np.mean(err)
                Wd -= float(learning_rate) * dW
                bd -= float(learning_rate) * float(db)
            else:
                ad, dad = _binary_activation_and_grad(binary_activation, zd, float(alpha_leaky))
                dz = (2.0 / X_train.shape[0]) * (ad - y_tr_d) * dad
                dW = X_train.T @ dz
                db = np.sum(dz)
                Wd -= float(learning_rate) * dW
                bd -= float(learning_rate) * float(db)

        out = _binary_activation_forward(binary_activation, X_test @ Wd + bd, float(alpha_leaky))
        if binary_activation == "tanh":
            pred = (out >= 0.0).astype(np.float32)
        else:
            pred = (out >= 0.5).astype(np.float32)

        per_digit_binary_accuracy.append(float(np.mean(pred == y_te_d)))
        per_digit_outputs.append(float(np.mean(out)))

    _LAST_MODELS.clear()
    _LAST_MODELS.update(
        {
            "binary_activation": binary_activation,
            "target_digit": int(target_digit),
            "alpha_leaky": float(alpha_leaky),
            "binary_W": Wb,
            "binary_b": float(bb),
            "multiclass_W": Wm,
            "multiclass_b": bm,
        }
    )

    return {
        "binary_activation": binary_activation,
        "target_digit": int(target_digit),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "binary_train_loss": binary_train_loss,
        "binary_test_loss": binary_test_loss,
        "binary_train_acc": binary_train_acc,
        "binary_test_acc": binary_test_acc,
        "multiclass_train_loss": multi_train_loss,
        "multiclass_test_loss": multi_test_loss,
        "multiclass_train_acc": multi_train_acc,
        "multiclass_test_acc": multi_test_acc,
        "multiclass_confusion": confusion.tolist(),
        "per_digit_binary_accuracy": per_digit_binary_accuracy,
        "per_digit_binary_output_mean": per_digit_outputs,
    }


def predict_uploaded_image(image_base64: str) -> dict[str, Any]:
    if "multiclass_W" not in _LAST_MODELS:
        raise ValueError("Train the model first before predicting an uploaded image.")

    vec = _decode_image_to_vector(image_base64)

    multi_logits = vec @ _LAST_MODELS["multiclass_W"] + _LAST_MODELS["multiclass_b"]
    multi_probs = _softmax(multi_logits)
    multi_pred = int(np.argmax(multi_probs, axis=1)[0])

    bin_z = vec @ _LAST_MODELS["binary_W"] + _LAST_MODELS["binary_b"]
    bin_out = _binary_activation_forward(
        _LAST_MODELS["binary_activation"],
        bin_z,
        float(_LAST_MODELS["alpha_leaky"]),
    )

    target_digit = int(_LAST_MODELS["target_digit"])
    if _LAST_MODELS["binary_activation"] == "tanh":
        bin_pred = 1 if float(bin_out[0, 0]) >= 0.0 else 0
    else:
        bin_pred = 1 if float(bin_out[0, 0]) >= 0.5 else 0

    one_vs_rest_scores: list[float] = []
    for d in range(10):
        # proxy score from multiclass probabilities to compare all digits in prediction view
        one_vs_rest_scores.append(float(multi_probs[0, d]))

    return {
        "predicted_digit_multiclass": multi_pred,
        "multiclass_probabilities": [float(v) for v in multi_probs[0]],
        "binary_target_digit": target_digit,
        "binary_output": float(bin_out[0, 0]),
        "binary_pred_label": int(bin_pred),
        "binary_statement": f"digit == {target_digit}" if bin_pred == 1 else f"digit != {target_digit}",
        "per_digit_output": one_vs_rest_scores,
    }


def _load_mnist() -> tuple[np.ndarray, np.ndarray]:
    global _MNIST_CACHE
    if _MNIST_CACHE is not None:
        return _MNIST_CACHE

    ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = ds.data.astype(np.float32) / 255.0
    y = ds.target.astype(np.int32)

    _MNIST_CACHE = (X, y)
    return _MNIST_CACHE


def _binary_activation_and_grad(name: str, z: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if name == "sigmoid":
        a = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return a, a * (1.0 - a)
    if name == "tanh":
        a = np.tanh(z)
        return a, 1.0 - a * a
    if name == "relu":
        a = np.maximum(0.0, z)
        return a, (z > 0).astype(np.float32)
    if name == "leaky_relu":
        a = np.where(z > 0, z, alpha * z)
        return a, np.where(z > 0, 1.0, alpha).astype(np.float32)
    raise ValueError("binary_step is not differentiable and handled separately")


def _binary_activation_forward(name: str, z: np.ndarray, alpha: float) -> np.ndarray:
    if name == "binary_step":
        return (z >= 0).astype(np.float32)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "leaky_relu":
        return np.where(z > 0, z, alpha * z)
    raise ValueError(f"Unknown activation '{name}'")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _decode_image_to_vector(image_base64: str) -> np.ndarray:
    raw = str(image_base64 or "").strip()
    if not raw:
        raise ValueError("image_base64 is required.")

    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]

    image_bytes = base64.b64decode(raw)
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.asarray(image, dtype=np.float32) / 255.0

    # Auto contrast direction: MNIST is white foreground on dark background.
    if float(arr.mean()) > 0.5:
        arr = 1.0 - arr

    arr = np.clip(arr, 0.0, 1.0)
    vec = arr.reshape(1, -1)
    return vec
