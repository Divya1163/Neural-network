"""Manual MLP training engine with notebook-style preprocessing and metrics."""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn import datasets as sk_datasets
except Exception:  # pragma: no cover - optional dependency fallback
    sk_datasets = None


BUILTIN_DATASETS: dict[str, str] = {
    "iris": "sklearn:iris",
    "wine": "sklearn:wine",
    "breast_cancer": "sklearn:breast_cancer",
    "digits": "sklearn:digits",
    "xor": "x1,x2,target\n0,0,0\n0,1,1\n1,0,1\n1,1,0",
}


def list_builtin_datasets() -> list[dict[str, str]]:
    return [
        {"name": "iris", "label": "Iris", "source": "sklearn"},
        {"name": "wine", "label": "Wine", "source": "sklearn"},
        {"name": "breast_cancer", "label": "Breast Cancer", "source": "sklearn"},
        {"name": "digits", "label": "Digits", "source": "sklearn"},
        {"name": "xor", "label": "XOR", "source": "manual"},
    ]


def get_dataset_csv(dataset_name: str) -> str:
    key = str(dataset_name or "").strip().lower()
    if key not in BUILTIN_DATASETS:
        supported = ", ".join(sorted(BUILTIN_DATASETS.keys()))
        raise ValueError(f"Unknown MLP dataset '{dataset_name}'. Supported: {supported}")

    raw = BUILTIN_DATASETS[key]
    if raw.startswith("sklearn:"):
        if sk_datasets is None:
            raise ValueError("scikit-learn is required for this built-in dataset. Install scikit-learn to continue.")

        dataset_key = raw.split(":", 1)[1]
        if dataset_key == "iris":
            ds = sk_datasets.load_iris(as_frame=True)
        elif dataset_key == "wine":
            ds = sk_datasets.load_wine(as_frame=True)
        elif dataset_key == "breast_cancer":
            ds = sk_datasets.load_breast_cancer(as_frame=True)
        elif dataset_key == "digits":
            ds = sk_datasets.load_digits(as_frame=True)
        else:
            raise ValueError(f"Unsupported sklearn dataset key '{dataset_key}'.")

        frame = ds.frame.copy()
        if "target" not in frame.columns:
            frame["target"] = ds.target
        return frame.to_csv(index=False)

    return raw


def train_from_csv(
    dataset_csv: str,
    target_column: str | None = None,
    selected_feature_columns: list[str] | None = None,
    row_threshold_for_drop: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.01,
    epochs: int = 250,
    batch_size: int = 32,
    hidden_layers_override: list[int] | None = None,
) -> dict[str, Any]:
    if not str(dataset_csv or "").strip():
        raise ValueError("dataset_csv is required.")

    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if not (0 < test_size < 0.9):
        raise ValueError("test_size must be between 0 and 0.9.")

    df = pd.read_csv(StringIO(dataset_csv))
    if df.empty:
        raise ValueError("Dataset is empty.")

    rows_before = int(len(df))
    null_count_before = int(df.isna().sum().sum())
    duplicate_count_before = int(df.duplicated().sum())

    if rows_before > row_threshold_for_drop:
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        cleaning_strategy = "Dropped null rows + dropped duplicates (large dataset strategy)"
    else:
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_values = df[col].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "missing"
                    df[col] = df[col].fillna(fill_value)
        df = df.drop_duplicates().reset_index(drop=True)
        cleaning_strategy = "Imputed nulls (median/mode) + dropped duplicates (small dataset strategy)"

    if len(df) < 4:
        raise ValueError("Dataset needs at least 4 rows after cleaning.")

    all_columns = df.columns.tolist()

    if not target_column:
        fallback_targets = ["target", "label", "class", "y"]
        target_column = next((col for col in fallback_targets if col in all_columns), all_columns[-1])

    if target_column not in all_columns:
        raise ValueError(f"target_column '{target_column}' was not found in dataset columns.")

    if selected_feature_columns:
        missing = [col for col in selected_feature_columns if col not in all_columns]
        if missing:
            raise ValueError(f"Missing selected_feature_columns: {missing}")
        feature_columns = [col for col in selected_feature_columns if col != target_column]
    else:
        feature_columns = [col for col in all_columns if col != target_column]

    if not feature_columns:
        raise ValueError("No feature columns available after selection.")

    X_raw = df[feature_columns].copy()
    y_raw = df[target_column].copy().astype(str)

    numeric_columns = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X_raw.columns if col not in numeric_columns]

    if numeric_columns:
        X_raw[numeric_columns] = X_raw[numeric_columns].fillna(X_raw[numeric_columns].median())
    if categorical_columns:
        for col in categorical_columns:
            mode_values = X_raw[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "missing"
            X_raw[col] = X_raw[col].fillna(fill_value)

    if categorical_columns:
        X_processed_df = pd.get_dummies(X_raw, columns=categorical_columns, drop_first=False)
    else:
        X_processed_df = X_raw.copy()

    if X_processed_df.shape[1] < 1:
        raise ValueError("Processed feature matrix is empty.")

    numeric_processed = X_processed_df.select_dtypes(include=[np.number]).columns.tolist()
    mean_map = {}
    std_map = {}
    if numeric_processed:
        for col in numeric_processed:
            col_mean = float(X_processed_df[col].mean())
            col_std = float(X_processed_df[col].std(ddof=0))
            if col_std == 0:
                col_std = 1.0
            mean_map[col] = col_mean
            std_map[col] = col_std
            X_processed_df[col] = (X_processed_df[col] - col_mean) / col_std

    X_processed = X_processed_df.to_numpy(dtype=float)

    class_names = sorted(y_raw.unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y_encoded = np.array([class_to_idx[value] for value in y_raw], dtype=int)

    n_samples = X_processed.shape[0]
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("Target column must have at least 2 classes.")

    train_idx, test_idx = _stratified_split_indices(y_encoded, test_size=test_size, random_state=random_state)

    X_train = X_processed[train_idx]
    X_test = X_processed[test_idx]
    y_train_int = y_encoded[train_idx]
    y_test_int = y_encoded[test_idx]

    input_dim = X_train.shape[1]

    if n_classes == 2:
        y_train = y_train_int.reshape(-1, 1).astype(float)
        y_test = y_test_int.reshape(-1, 1).astype(float)
    else:
        y_train = np.eye(n_classes, dtype=float)[y_train_int]
        y_test = np.eye(n_classes, dtype=float)[y_test_int]

    if hidden_layers_override:
        hidden_layers = [int(v) for v in hidden_layers_override]
        if any(v < 1 for v in hidden_layers):
            raise ValueError("hidden_layers_override must contain positive integers.")
    else:
        h1 = int(np.clip(input_dim * 2, 16, 128))
        h2 = int(np.clip(input_dim, 8, 64))
        hidden_layers = [h1, h2]

    output_dim = 1 if n_classes == 2 else n_classes
    layer_dims = [input_dim] + hidden_layers + [output_dim]

    rng = np.random.default_rng(seed=random_state)
    W: list[np.ndarray] = []
    b: list[np.ndarray] = []
    for i in range(len(layer_dims) - 1):
        fan_in = layer_dims[i]
        fan_out = layer_dims[i + 1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
        b.append(np.zeros((1, fan_out), dtype=float))

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    step_log: list[dict[str, Any]] = []

    n_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(X_train.shape[0])
        X_epoch = X_train[perm]
        y_epoch = y_train[perm]

        epoch_loss_sum = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, X_train.shape[0])
            X_batch = X_epoch[start:end]
            y_batch = y_epoch[start:end]

            activations = [X_batch]
            z_values = []

            for i in range(len(hidden_layers)):
                z_i = activations[-1] @ W[i] + b[i]
                a_i = np.maximum(0.0, z_i)
                z_values.append(z_i)
                activations.append(a_i)

            z_out = activations[-1] @ W[-1] + b[-1]
            z_values.append(z_out)

            if n_classes == 2:
                probs = 1.0 / (1.0 + np.exp(-np.clip(z_out, -30, 30)))
                eps = 1e-12
                batch_loss = -np.mean(y_batch * np.log(probs + eps) + (1.0 - y_batch) * np.log(1.0 - probs + eps))
            else:
                shifted = z_out - np.max(z_out, axis=1, keepdims=True)
                exp_scores = np.exp(shifted)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                eps = 1e-12
                batch_loss = -np.mean(np.sum(y_batch * np.log(probs + eps), axis=1))

            activations.append(probs)
            epoch_loss_sum += float(batch_loss) * X_batch.shape[0]

            grads_W: list[np.ndarray] = [np.zeros_like(weights) for weights in W]
            grads_b: list[np.ndarray] = [np.zeros_like(bias) for bias in b]

            dZ = activations[-1] - y_batch
            grads_W[-1] = (activations[-2].T @ dZ) / X_batch.shape[0]
            grads_b[-1] = np.mean(dZ, axis=0, keepdims=True)

            for i in range(len(hidden_layers) - 1, -1, -1):
                dA_prev = dZ @ W[i + 1].T
                relu_grad = (z_values[i] > 0).astype(float)
                dZ = dA_prev * relu_grad
                grads_W[i] = (activations[i].T @ dZ) / X_batch.shape[0]
                grads_b[i] = np.mean(dZ, axis=0, keepdims=True)

            if epoch == 1 and batch_idx == 0:
                grad_norms = [float(np.linalg.norm(g)) for g in grads_W]
                step_log.append(
                    {
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "batch_size": int(X_batch.shape[0]),
                        "train_loss": float(batch_loss),
                        "gradient_norms": grad_norms,
                        "sample_probabilities": [float(v) for v in np.ravel(probs)[: min(6, probs.size)]],
                    }
                )

            for i in range(len(W)):
                W[i] = W[i] - learning_rate * grads_W[i]
                b[i] = b[i] - learning_rate * grads_b[i]

        train_loss_epoch = epoch_loss_sum / X_train.shape[0]

        train_probs = _forward_predict(X_train, W, b, n_classes)
        test_probs = _forward_predict(X_test, W, b, n_classes)

        train_loss_eval = _compute_loss(train_probs, y_train, n_classes)
        test_loss_eval = _compute_loss(test_probs, y_test, n_classes)

        if n_classes == 2:
            y_train_pred = (train_probs >= 0.5).astype(int).ravel()
            y_test_pred = (test_probs >= 0.5).astype(int).ravel()
            train_confidence = train_probs.ravel()
            test_confidence = test_probs.ravel()
        else:
            y_train_pred = np.argmax(train_probs, axis=1)
            y_test_pred = np.argmax(test_probs, axis=1)
            train_confidence = np.max(train_probs, axis=1)
            test_confidence = np.max(test_probs, axis=1)

        train_acc = float(np.mean(y_train_pred == y_train_int))
        test_acc = float(np.mean(y_test_pred == y_test_int))

        train_loss_history.append(float(train_loss_eval if np.isfinite(train_loss_eval) else train_loss_epoch))
        test_loss_history.append(float(test_loss_eval))
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if epoch in {1, epochs} or epoch % max(1, epochs // 10) == 0:
            step_log.append(
                {
                    "epoch": epoch,
                    "summary": (
                        f"Epoch {epoch}/{epochs}: train_loss={train_loss_history[-1]:.5f}, "
                        f"test_loss={test_loss_history[-1]:.5f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
                    ),
                }
            )

    if n_classes == 2:
        final_train_pred = (train_probs >= 0.5).astype(int).ravel()
        final_test_pred = (test_probs >= 0.5).astype(int).ravel()
    else:
        final_train_pred = np.argmax(train_probs, axis=1)
        final_test_pred = np.argmax(test_probs, axis=1)

    cm = _confusion_matrix(y_test_int, final_test_pred, n_classes)
    cls_report = _classification_report(cm, class_names)

    predictions_preview = []
    limit = min(40, X_test.shape[0])
    for i in range(limit):
        predictions_preview.append(
            {
                "row_index": int(test_idx[i]),
                "true_class": int(y_test_int[i]),
                "pred_class": int(final_test_pred[i]),
                "true_class_name": str(class_names[int(y_test_int[i])]),
                "pred_class_name": str(class_names[int(final_test_pred[i])]),
                "confidence": float(test_confidence[i]),
            }
        )

    pipeline_steps = [
        "Loaded CSV into DataFrame",
        cleaning_strategy,
        "Selected target/feature columns",
        "Applied one-hot encoding for categorical inputs",
        "Standardized processed numeric features",
        "Performed stratified train/test split",
        "Auto-designed hidden layers (or used override)",
        "Ran manual forward/backward training loop",
        "Evaluated final metrics and confusion matrix",
    ]

    final_params = {
        "weights": [weights.tolist() for weights in W],
        "biases": [bias.tolist() for bias in b],
        "layer_dims": layer_dims,
    }

    return {
        "rows_before": rows_before,
        "rows_after": int(len(df)),
        "null_count_before": null_count_before,
        "null_count_after": int(df.isna().sum().sum()),
        "duplicate_count_before": duplicate_count_before,
        "duplicate_count_after": int(df.duplicated().sum()),
        "cleaning_strategy": cleaning_strategy,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "processed_feature_count": int(X_processed.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "class_names": [str(c) for c in class_names],
        "n_classes": int(n_classes),
        "layer_dims": layer_dims,
        "hidden_layers": hidden_layers,
        "epochs": int(epochs),
        "epochs_trained": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "train_acc_history": train_acc_history,
        "test_acc_history": test_acc_history,
        "final_train_accuracy": float(train_acc_history[-1]),
        "final_test_accuracy": float(test_acc_history[-1]),
        "final_test_loss": float(test_loss_history[-1]),
        "step_log": step_log,
        "pipeline_steps": pipeline_steps,
        "predictions_preview": predictions_preview,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "final_params": final_params,
        "processing_stats": {
            "scaled_columns": numeric_processed,
            "mean_map": mean_map,
            "std_map": std_map,
        },
    }


def _stratified_split_indices(y: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=random_state)
    unique_classes = np.unique(y)

    train_parts = []
    test_parts = []

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        if n_test >= len(cls_idx):
            n_test = len(cls_idx) - 1
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx


def _forward_predict(X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray], n_classes: int) -> np.ndarray:
    activations = X
    hidden_count = len(weights) - 1

    for i in range(hidden_count):
        activations = np.maximum(0.0, activations @ weights[i] + biases[i])

    logits = activations @ weights[-1] + biases[-1]

    if n_classes == 2:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _compute_loss(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    eps = 1e-12
    if n_classes == 2:
        return float(-np.mean(y_true * np.log(probs + eps) + (1.0 - y_true) * np.log(1.0 - probs + eps)))
    return float(-np.mean(np.sum(y_true * np.log(probs + eps), axis=1)))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def _classification_report(cm: np.ndarray, class_names: list[str]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    total = int(np.sum(cm))
    diag = int(np.trace(cm))

    for idx, class_name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fp = float(np.sum(cm[:, idx]) - tp)
        fn = float(np.sum(cm[idx, :]) - tp)
        support = float(np.sum(cm[idx, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[str(class_name)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }

    accuracy = diag / total if total > 0 else 0.0
    report["accuracy"] = {"value": round(float(accuracy), 4)}
    return report
