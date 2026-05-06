"""Core perceptron logic used by the Flask backend."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

BUILTIN_DATASETS: Dict[str, str] = {
    "and": "x1,x2,label\n0,0,0\n0,1,0\n1,0,0\n1,1,1",
    "or": "x1,x2,label\n0,0,0\n0,1,1\n1,0,1\n1,1,1",
    "nand": "x1,x2,label\n0,0,1\n0,1,1\n1,0,1\n1,1,0",
}


def get_dataset_csv(gate_name: str) -> str:
    key = gate_name.lower()
    if key not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown built-in gate dataset: {gate_name}")
    return BUILTIN_DATASETS[key]


def _parse_csv_rows(raw_csv: str) -> Tuple[List[str], List[List[float]]]:
    if not raw_csv or not raw_csv.strip():
        raise ValueError("Training data cannot be empty.")

    lines = [line.strip() for line in raw_csv.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Provide at least one header/data line and one data row.")

    first_tokens = [token.strip() for token in lines[0].split(",")]
    has_header = not all(_is_finite_number(token) for token in first_tokens)

    if has_header:
        headers = first_tokens
        data_lines = lines[1:]
    else:
        data_lines = lines
        width = len(first_tokens)
        headers = [f"x{i + 1}" for i in range(width - 1)] + ["label"]

    rows: List[List[float]] = []
    for row_idx, line in enumerate(data_lines, start=1):
        tokens = [token.strip() for token in line.split(",")]
        parsed_row: List[float] = []

        for col_idx, token in enumerate(tokens, start=1):
            if not _is_finite_number(token):
                raise ValueError(f"Non-numeric value at row {row_idx}, col {col_idx}.")
            parsed_row.append(float(token))

        rows.append(parsed_row)

    _validate_rows(rows)
    return headers, rows


def _validate_rows(rows: Sequence[Sequence[float]]) -> None:
    if not rows:
        raise ValueError("No data rows found.")

    width = len(rows[0])
    if width < 2:
        raise ValueError("Each row needs at least one feature and one label.")

    for row_idx, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(f"Column mismatch at row {row_idx}.")

        label = row[-1]
        if label not in (0.0, 1.0):
            raise ValueError(f"Label at row {row_idx} must be 0 or 1.")


def parse_initial_weights(raw_weights: Sequence[float] | str, feature_count: int) -> List[float]:
    if isinstance(raw_weights, str):
        parts = [p.strip() for p in raw_weights.split(",") if p.strip()]
        values: List[float] = []
        for idx, token in enumerate(parts, start=1):
            if not _is_finite_number(token):
                raise ValueError(f"Initial weight at position {idx} is not numeric.")
            values.append(float(token))
    else:
        values = [float(v) for v in raw_weights]

    if len(values) != feature_count:
        raise ValueError(f"Initial weights must have exactly {feature_count} values.")

    return values


def train_from_csv(
    dataset_csv: str,
    learning_rate: float,
    epochs: int,
    initial_weights: Sequence[float] | str,
    initial_bias: float,
) -> Dict[str, object]:
    headers, rows = _parse_csv_rows(dataset_csv)

    feature_count = len(rows[0]) - 1
    x_data = [row[:feature_count] for row in rows]
    y_data = [int(row[-1]) for row in rows]

    weights = parse_initial_weights(initial_weights, feature_count)
    bias = float(initial_bias)

    epoch_history: List[Dict[str, object]] = []
    walkthrough: List[Dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        total_error = 0

        for sample_idx, (x_row, y_true) in enumerate(zip(x_data, y_data), start=1):
            z_value = _dot(weights, x_row) + bias
            prediction = 1 if z_value >= 0 else 0
            error = y_true - prediction

            delta_weights = [learning_rate * error * x_row[i] for i in range(feature_count)]
            for i in range(feature_count):
                weights[i] += delta_weights[i]

            delta_bias = learning_rate * error
            bias += delta_bias
            total_error += abs(error)

            if epoch == 1:
                walkthrough.append(
                    {
                        "sample_index": sample_idx,
                        "sample": list(x_row),
                        "label": y_true,
                        "z": z_value,
                        "prediction": prediction,
                        "error": error,
                        "delta_weights": list(delta_weights),
                        "delta_bias": delta_bias,
                        "new_weights": list(weights),
                        "new_bias": bias,
                    }
                )

        epoch_history.append(
            {
                "epoch": epoch,
                "total_error": total_error,
                "weights": list(weights),
                "bias": bias,
            }
        )

        if total_error == 0:
            break

    converged = epoch_history[-1]["total_error"] == 0
    return {
        "headers": headers,
        "feature_count": feature_count,
        "final_weights": list(weights),
        "final_bias": bias,
        "epoch_history": epoch_history,
        "walkthrough": walkthrough,
        "epochs_trained": len(epoch_history),
        "converged": converged,
    }


def predict_from_model(weights: Sequence[float], bias: float, inputs: Sequence[float]) -> Dict[str, float | int]:
    if len(weights) != len(inputs):
        raise ValueError("Input length must match number of model weights.")

    z_value = _dot(weights, inputs) + bias
    prediction = 1 if z_value >= 0 else 0
    return {"z": z_value, "prediction": prediction}


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    total = 0.0
    for lv, rv in zip(left, right):
        total += lv * rv
    return total


def _is_finite_number(text: str) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return value == value and value not in (float("inf"), float("-inf"))
