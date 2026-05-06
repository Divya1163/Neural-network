"""Manual activation function analytics for interactive UI."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTIVATION_LIBRARY: dict[str, dict[str, Any]] = {
    "binary_step": {
        "label": "Binary Step",
        "level": "Beginner",
        "category": "Threshold",
        "best_use": "Perceptron-style hard decisions and logic gates.",
        "impact": "Creates a sharp cutoff and non-probabilistic output (0 or 1).",
        "limitations": "Not differentiable; unsuitable for gradient descent.",
    },
    "linear": {
        "label": "Linear",
        "level": "Beginner",
        "category": "Identity",
        "best_use": "Output layer for regression with unrestricted value range.",
        "impact": "Preserves sign/magnitude directly from pre-activation.",
        "limitations": "No nonlinearity, so representational power is limited.",
    },
    "sigmoid": {
        "label": "Sigmoid",
        "level": "Beginner",
        "category": "Probabilistic",
        "best_use": "Binary classification output layers.",
        "impact": "Maps outputs to [0, 1] for probability-like interpretation.",
        "limitations": "Saturates at extremes and can cause vanishing gradients.",
    },
    "tanh": {
        "label": "Tanh",
        "level": "Intermediate",
        "category": "Zero-centered",
        "best_use": "Hidden layers when zero-centered activations help optimization.",
        "impact": "Range [-1, 1] often gives smoother optimization than sigmoid.",
        "limitations": "Still saturates for large absolute input values.",
    },
    "relu": {
        "label": "ReLU",
        "level": "Intermediate",
        "category": "Piecewise",
        "best_use": "Default hidden activation in many deep architectures.",
        "impact": "Sparse activation and stable positive-side gradients.",
        "limitations": "Dead neurons possible for long negative-side exposure.",
    },
    "leaky_relu": {
        "label": "Leaky ReLU",
        "level": "Intermediate",
        "category": "Piecewise",
        "best_use": "Hidden layers where dead ReLU risk is a concern.",
        "impact": "Keeps a small gradient on negative side for better recovery.",
        "limitations": "Alpha tuning needed and interpretation less intuitive.",
    },
    "elu": {
        "label": "ELU",
        "level": "Advanced",
        "category": "Smooth piecewise",
        "best_use": "Hidden layers needing smooth negative saturation.",
        "impact": "Can speed learning with nonzero negative output mean.",
        "limitations": "More expensive than ReLU and alpha-sensitive.",
    },
    "softplus": {
        "label": "Softplus",
        "level": "Advanced",
        "category": "Smooth ReLU",
        "best_use": "When a smooth ReLU-like activation is preferred.",
        "impact": "Differentiable everywhere and avoids hard kink at zero.",
        "limitations": "May produce less sparse activations than ReLU.",
    },
    "swish": {
        "label": "Swish",
        "level": "Advanced",
        "category": "Self-gated",
        "best_use": "Deep networks where smoother gradients can help.",
        "impact": "Non-monotonic smooth behavior can improve optimization.",
        "limitations": "Slightly heavier computation than ReLU-like functions.",
    },
    "softmax": {
        "label": "Softmax",
        "level": "Intermediate",
        "category": "Multiclass output",
        "best_use": "Output layer for mutually exclusive multiclass classification.",
        "impact": "Converts logits to class probabilities summing to 1.",
        "limitations": "Used at output only; not a hidden-layer default.",
    },
}


def list_activation_cards() -> list[dict[str, str]]:
    cards = []
    for name, info in ACTIVATION_LIBRARY.items():
        cards.append(
            {
                "name": name,
                "label": str(info["label"]),
                "level": str(info["level"]),
                "category": str(info["category"]),
                "best_use": str(info["best_use"]),
            }
        )
    return cards


def analyze_activation(
    activation_name: str,
    weight: float = 1.0,
    bias: float = 0.0,
    alpha: float = 0.01,
    x_min: float = -6.0,
    x_max: float = 6.0,
    points: int = 241,
    input_values: list[float] | None = None,
) -> dict[str, Any]:
    key = str(activation_name or "").strip().lower()
    if key not in ACTIVATION_LIBRARY:
        supported = ", ".join(sorted(ACTIVATION_LIBRARY.keys()))
        raise ValueError(f"Unsupported activation '{activation_name}'. Supported: {supported}")

    if points < 21:
        points = 21

    x_values = np.linspace(float(x_min), float(x_max), int(points))

    if key == "softmax":
        z1 = float(weight) * x_values + float(bias)
        z2 = np.zeros_like(z1)

        stacked = np.stack([z1, z2], axis=1)
        shifted = stacked - np.max(stacked, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        curve_y = probs[:, 0]
        derivative = curve_y * (1.0 - curve_y)
    else:
        z_values = float(weight) * x_values + float(bias)
        curve_y, derivative = _single_activation_and_derivative(key, z_values, float(alpha))

    if input_values is None or len(input_values) == 0:
        input_values = [-3.0, -1.0, 0.0, 1.0, 3.0]

    table_rows: list[dict[str, float]] = []
    for v in input_values:
        x_scalar = float(v)
        if key == "softmax":
            z1 = float(weight) * x_scalar + float(bias)
            z2 = 0.0
            shifted1 = z1 - max(z1, z2)
            shifted2 = z2 - max(z1, z2)
            e1 = np.exp(shifted1)
            e2 = np.exp(shifted2)
            p1 = float(e1 / (e1 + e2))
            d1 = float(p1 * (1 - p1))
            table_rows.append(
                {
                    "input": x_scalar,
                    "pre_activation": z1,
                    "activation": p1,
                    "derivative": d1,
                }
            )
        else:
            z = float(weight) * x_scalar + float(bias)
            a, d = _single_activation_and_derivative(key, np.array([z]), float(alpha))
            table_rows.append(
                {
                    "input": x_scalar,
                    "pre_activation": z,
                    "activation": float(a[0]),
                    "derivative": float(d[0]),
                }
            )

    toy_dataset = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    if key == "softmax":
        z1_ds = float(weight) * toy_dataset + float(bias)
        z2_ds = np.zeros_like(z1_ds)
        ds_stack = np.stack([z1_ds, z2_ds], axis=1)
        ds_shift = ds_stack - np.max(ds_stack, axis=1, keepdims=True)
        ds_exp = np.exp(ds_shift)
        ds_prob = ds_exp / np.sum(ds_exp, axis=1, keepdims=True)
        impact_rows = [
            {
                "feature": float(toy_dataset[i]),
                "class1_probability": float(ds_prob[i, 0]),
                "class2_probability": float(ds_prob[i, 1]),
            }
            for i in range(toy_dataset.shape[0])
        ]
    else:
        z_ds = float(weight) * toy_dataset + float(bias)
        a_ds, _ = _single_activation_and_derivative(key, z_ds, float(alpha))
        impact_rows = [
            {
                "feature": float(toy_dataset[i]),
                "activated_output": float(a_ds[i]),
            }
            for i in range(toy_dataset.shape[0])
        ]

    info = ACTIVATION_LIBRARY[key]

    return {
        "name": key,
        "label": info["label"],
        "level": info["level"],
        "category": info["category"],
        "best_use": info["best_use"],
        "impact": info["impact"],
        "limitations": info["limitations"],
        "x_values": [float(v) for v in x_values],
        "y_values": [float(v) for v in curve_y],
        "derivative_values": [float(v) for v in derivative],
        "sample_table": table_rows,
        "impact_table": impact_rows,
        "parameters": {
            "weight": float(weight),
            "bias": float(bias),
            "alpha": float(alpha),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "points": int(points),
        },
    }


def _single_activation_and_derivative(name: str, z: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if name == "binary_step":
        y = (z >= 0).astype(float)
        d = np.zeros_like(y)
        return y, d

    if name == "linear":
        y = z
        d = np.ones_like(z)
        return y, d

    if name == "sigmoid":
        y = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        d = y * (1.0 - y)
        return y, d

    if name == "tanh":
        y = np.tanh(z)
        d = 1.0 - y * y
        return y, d

    if name == "relu":
        y = np.maximum(0.0, z)
        d = (z > 0).astype(float)
        return y, d

    if name == "leaky_relu":
        y = np.where(z > 0.0, z, alpha * z)
        d = np.where(z > 0.0, 1.0, alpha)
        return y, d

    if name == "elu":
        y = np.where(z > 0.0, z, alpha * (np.exp(z) - 1.0))
        d = np.where(z > 0.0, 1.0, alpha * np.exp(z))
        return y, d

    if name == "softplus":
        y = np.log1p(np.exp(z))
        d = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        return y, d

    if name == "swish":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        y = z * sig
        d = sig + z * sig * (1.0 - sig)
        return y, d

    raise ValueError(f"Derivative implementation missing for activation '{name}'.")
