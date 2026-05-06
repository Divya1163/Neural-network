"""Backpropagation engine for a tiny feedforward network (single hidden layer)."""

from __future__ import annotations

import math
from typing import Any


BUILTIN_DATASETS = {
    "xor": "x1,x2,label\n0,0,0\n0,1,1\n1,0,1\n1,1,0",
}


def get_dataset_csv(dataset_name: str) -> str:
    key = str(dataset_name or "").strip().lower()
    if key not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown backprop dataset '{dataset_name}'. Supported: {', '.join(BUILTIN_DATASETS.keys())}")
    return BUILTIN_DATASETS[key]


def _parse_csv_dataset(dataset_csv: str) -> tuple[list[str], list[list[float]], list[float]]:
    lines = [line.strip() for line in str(dataset_csv).replace("\r\n", "\n").split("\n") if line.strip()]
    if len(lines) < 2:
        raise ValueError("Dataset must include header and at least one data row.")

    headers = [h.strip() for h in lines[0].split(",")]
    if len(headers) < 2:
        raise ValueError("Dataset must include at least one feature and one label column.")

    rows: list[list[float]] = []
    labels: list[float] = []

    for row_index, line in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(headers):
            raise ValueError(
                f"Row {row_index} has {len(parts)} values but expected {len(headers)} based on header."
            )

        numeric = [float(value) for value in parts]
        rows.append(numeric[:-1])
        labels.append(float(numeric[-1]))

    return headers, rows, labels


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def train_from_csv(
    dataset_csv: str,
    learning_rate: float,
    epochs: int,
    hidden_neurons: int,
    initial_input_hidden: list[list[float]] | None = None,
    initial_hidden_output: list[float] | None = None,
    initial_hidden_bias: list[float] | None = None,
    initial_output_bias: float | None = None,
) -> dict[str, Any]:
    headers, features, labels = _parse_csv_dataset(dataset_csv)

    if not features:
        raise ValueError("Dataset has no training rows.")

    feature_count = len(features[0])
    if feature_count < 1:
        raise ValueError("Dataset must include at least one feature column.")

    if hidden_neurons < 1:
        raise ValueError("hidden_neurons must be at least 1.")

    for row in features:
        if len(row) != feature_count:
            raise ValueError("All rows must have the same number of features.")

    if initial_input_hidden and len(initial_input_hidden) != hidden_neurons:
        raise ValueError("initial_input_hidden row count must match hidden_neurons.")

    if initial_hidden_output and len(initial_hidden_output) != hidden_neurons:
        raise ValueError("initial_hidden_output length must match hidden_neurons.")

    if initial_hidden_bias and len(initial_hidden_bias) != hidden_neurons:
        raise ValueError("initial_hidden_bias length must match hidden_neurons.")

    # Initialize weights deterministically when not provided.
    weights_input_hidden: list[list[float]] = []
    for h in range(hidden_neurons):
        if initial_input_hidden:
            row = [float(v) for v in initial_input_hidden[h]]
            if len(row) != feature_count:
                raise ValueError("Each initial_input_hidden row must match feature count.")
            weights_input_hidden.append(row)
        else:
            row = []
            for f in range(feature_count):
                row.append(0.10 + 0.05 * (h * feature_count + f + 1))
            weights_input_hidden.append(row)

    if initial_hidden_output:
        weights_hidden_output = [float(v) for v in initial_hidden_output]
    else:
        weights_hidden_output = [0.20 + 0.05 * (h + 1) for h in range(hidden_neurons)]

    if initial_hidden_bias:
        bias_hidden = [float(v) for v in initial_hidden_bias]
    else:
        bias_hidden = [0.35 for _ in range(hidden_neurons)]

    if initial_output_bias is not None:
        bias_output = float(initial_output_bias)
    else:
        bias_output = 0.60

    epoch_history: list[dict[str, Any]] = []
    walkthrough: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        total_squared_error = 0.0

        for sample_index, sample in enumerate(features, start=1):
            target = labels[sample_index - 1]

            hidden_nets: list[float] = []
            hidden_outputs: list[float] = []

            for h in range(hidden_neurons):
                net_h = bias_hidden[h]
                for f in range(feature_count):
                    net_h += sample[f] * weights_input_hidden[h][f]
                out_h = _sigmoid(net_h)
                hidden_nets.append(net_h)
                hidden_outputs.append(out_h)

            output_net = bias_output
            for h in range(hidden_neurons):
                output_net += hidden_outputs[h] * weights_hidden_output[h]
            output = _sigmoid(output_net)

            error_output = target - output
            total_squared_error += error_output * error_output

            delta_output = error_output * output * (1.0 - output)

            delta_hidden: list[float] = []
            for h in range(hidden_neurons):
                err_h = delta_output * weights_hidden_output[h]
                delta_h = err_h * hidden_outputs[h] * (1.0 - hidden_outputs[h])
                delta_hidden.append(delta_h)

            old_weights_hidden_output = weights_hidden_output[:]
            old_bias_output = bias_output
            old_weights_input_hidden = [row[:] for row in weights_input_hidden]
            old_bias_hidden = bias_hidden[:]

            delta_weights_hidden_output: list[float] = []
            for h in range(hidden_neurons):
                delta_w = float(learning_rate) * delta_output * hidden_outputs[h]
                weights_hidden_output[h] += delta_w
                delta_weights_hidden_output.append(delta_w)

            delta_bias_output = float(learning_rate) * delta_output
            bias_output += delta_bias_output

            delta_weights_input_hidden: list[list[float]] = []
            for h in range(hidden_neurons):
                delta_row: list[float] = []
                for f in range(feature_count):
                    delta_w = float(learning_rate) * delta_hidden[h] * sample[f]
                    weights_input_hidden[h][f] += delta_w
                    delta_row.append(delta_w)
                delta_weights_input_hidden.append(delta_row)

            delta_bias_hidden: list[float] = []
            for h in range(hidden_neurons):
                delta_b = float(learning_rate) * delta_hidden[h]
                bias_hidden[h] += delta_b
                delta_bias_hidden.append(delta_b)

            if epoch == 1:
                walkthrough.append(
                    {
                        "sample_index": sample_index,
                        "sample": [float(v) for v in sample],
                        "label": float(target),
                        "hidden_outputs": [float(v) for v in hidden_outputs],
                        "output": float(output),
                        "error": float(error_output),
                        "delta_output": float(delta_output),
                        "delta_hidden": [float(v) for v in delta_hidden],
                        "delta_weights_hidden_output": [float(v) for v in delta_weights_hidden_output],
                        "delta_bias_output": float(delta_bias_output),
                        "old_weights_hidden_output": [float(v) for v in old_weights_hidden_output],
                        "new_weights_hidden_output": [float(v) for v in weights_hidden_output],
                        "old_bias_output": float(old_bias_output),
                        "new_bias_output": float(bias_output),
                        "old_weights_input_hidden": [[float(v) for v in row] for row in old_weights_input_hidden],
                        "new_weights_input_hidden": [[float(v) for v in row] for row in weights_input_hidden],
                        "old_bias_hidden": [float(v) for v in old_bias_hidden],
                        "new_bias_hidden": [float(v) for v in bias_hidden],
                        "delta_weights_input_hidden": [[float(v) for v in row] for row in delta_weights_input_hidden],
                        "delta_bias_hidden": [float(v) for v in delta_bias_hidden],
                    }
                )

        mse = total_squared_error / len(features)

        epoch_history.append(
            {
                "epoch": epoch,
                "mse": float(mse),
                "weights_hidden_output": [float(v) for v in weights_hidden_output],
                "weights_input_hidden_flat": [
                    float(v)
                    for row in weights_input_hidden
                    for v in row
                ],
                "bias_hidden": [float(v) for v in bias_hidden],
                "bias_output": float(bias_output),
            }
        )

    predictions: list[dict[str, Any]] = []
    for idx, sample in enumerate(features):
        pred = predict_from_model(
            weights_input_hidden=weights_input_hidden,
            weights_hidden_output=weights_hidden_output,
            bias_hidden=bias_hidden,
            bias_output=bias_output,
            inputs=sample,
        )
        predictions.append(
            {
                "sample": [float(v) for v in sample],
                "target": float(labels[idx]),
                "probability": float(pred["probability"]),
                "class": int(pred["prediction"]),
            }
        )

    final_mse = epoch_history[-1]["mse"] if epoch_history else 0.0

    return {
        "headers": headers,
        "feature_count": feature_count,
        "hidden_neurons": hidden_neurons,
        "epochs_trained": int(epochs),
        "learning_rate": float(learning_rate),
        "epoch_history": epoch_history,
        "walkthrough": walkthrough,
        "predictions": predictions,
        "final_mse": float(final_mse),
        "converged": bool(final_mse < 0.05),
        "final_params": {
            "weights_input_hidden": [[float(v) for v in row] for row in weights_input_hidden],
            "weights_hidden_output": [float(v) for v in weights_hidden_output],
            "bias_hidden": [float(v) for v in bias_hidden],
            "bias_output": float(bias_output),
        },
    }


def predict_from_model(
    weights_input_hidden: list[list[float]],
    weights_hidden_output: list[float],
    bias_hidden: list[float],
    bias_output: float,
    inputs: list[float],
) -> dict[str, Any]:
    sample = [float(v) for v in inputs]
    if not weights_input_hidden:
        raise ValueError("weights_input_hidden is required.")

    feature_count = len(weights_input_hidden[0])
    if len(sample) != feature_count:
        raise ValueError(f"inputs length ({len(sample)}) must match feature_count ({feature_count}).")

    hidden_neurons = len(weights_input_hidden)

    if len(weights_hidden_output) != hidden_neurons:
        raise ValueError("weights_hidden_output length must match hidden layer size.")
    if len(bias_hidden) != hidden_neurons:
        raise ValueError("bias_hidden length must match hidden layer size.")

    hidden_outputs: list[float] = []
    for h in range(hidden_neurons):
        net_h = float(bias_hidden[h])
        for f in range(feature_count):
            net_h += sample[f] * float(weights_input_hidden[h][f])
        hidden_outputs.append(_sigmoid(net_h))

    output_net = float(bias_output)
    for h in range(hidden_neurons):
        output_net += hidden_outputs[h] * float(weights_hidden_output[h])

    probability = _sigmoid(output_net)
    prediction = 1 if probability >= 0.5 else 0

    return {
        "inputs": sample,
        "probability": float(probability),
        "prediction": int(prediction),
        "hidden_outputs": [float(v) for v in hidden_outputs],
    }
