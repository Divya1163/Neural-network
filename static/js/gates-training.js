// UI layer for perceptron page.
// Training and prediction logic is handled by Flask Python backend APIs.

const chartState = {};

const gateStates = {
    and: { model: null, featureCount: 2 },
    or: { model: null, featureCount: 2 },
    custom: { model: null, featureCount: 2 },
};

const defaultCustomDataset = [
    "x1,x2,label",
    "0,0,0",
    "0,1,0",
    "1,0,1",
    "1,1,1",
].join("\n");

document.addEventListener("DOMContentLoaded", async () => {
    wireEvents();
    preparePredictionInputs("and", 2);
    preparePredictionInputs("or", 2);
    preparePredictionInputs("custom", 2);

    await preloadBuiltInDataset("and");
    await preloadBuiltInDataset("or");

    element("customData").value = defaultCustomDataset;
});

function wireEvents() {
    document.querySelectorAll(".train-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const gate = btn.dataset.gate;
            trainGate(gate);
        });
    });

    document.querySelectorAll(".predict-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const gate = btn.dataset.gate;
            predictGate(gate);
        });
    });

    element("loadCustomSampleBtn").addEventListener("click", () => {
        element("customData").value = defaultCustomDataset;
        setStatus("customStatus", "Custom sample dataset loaded.", false);
    });
}

async function preloadBuiltInDataset(gate) {
    try {
        const response = await fetch(`/api/perceptron/dataset/${gate}`);
        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || `Could not load ${gate} dataset.`);
        }
        element(`${gate}Data`).value = payload.dataset_csv;
    } catch (error) {
        setStatus(`${gate}Status`, error.message, true);
    }
}

async function trainGate(gate) {
    const config = getGateConfig(gate);
    try {
        const requestBody = buildTrainRequest(gate);

        const response = await fetch("/api/perceptron/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Training failed.");
        }

        gateStates[gate] = {
            model: payload,
            featureCount: payload.feature_count,
        };

        preparePredictionInputs(gate, payload.feature_count);
        renderCalcWalkthrough(gate);
        renderEpochTable(gate);
        renderSummary(gate);
        renderChart(gate);

        setStatus(config.statusId, payload.converged ? "Training completed and converged." : "Training completed but not fully converged.", !payload.converged);
        setStatus(config.predictionStatusId, "Model trained. Enter inputs and click Predict.", false);
    } catch (error) {
        setStatus(config.statusId, error.message, true);
    }
}

function buildTrainRequest(gate) {
    const config = getGateConfig(gate);
    const learningRate = parseFloat(element(config.lrId).value);
    const epochs = parseInt(element(config.epochsId).value, 10);
    const initialBias = parseFloat(element(config.biasId).value);

    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
        throw new Error("Learning rate must be > 0 and <= 1.");
    }
    if (!Number.isInteger(epochs) || epochs < 1) {
        throw new Error("Epochs must be a positive integer.");
    }
    if (!Number.isFinite(initialBias)) {
        throw new Error("Initial bias must be a valid number.");
    }

    const datasetCsv = element(config.dataId).value.trim();
    if (!datasetCsv) {
        throw new Error("Training data is required.");
    }

    const initialWeights = parseWeightInput(element(config.weightsId).value);

    return {
        gate,
        dataset_csv: datasetCsv,
        learning_rate: learningRate,
        epochs,
        initial_weights: initialWeights,
        initial_bias: initialBias,
    };
}

async function predictGate(gate) {
    const config = getGateConfig(gate);
    const state = gateStates[gate];

    if (!state.model) {
        setStatus(config.predictionStatusId, "Train the model before prediction.", true);
        return;
    }

    const values = [];
    for (let i = 0; i < state.featureCount; i += 1) {
        const num = Number(element(`${gate}Pred${i}`).value);
        if (!Number.isFinite(num)) {
            setStatus(config.predictionStatusId, `Enter a valid numeric value for x${i + 1}.`, true);
            return;
        }
        values.push(num);
    }

    try {
        const response = await fetch("/api/perceptron/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                weights: state.model.final_weights,
                bias: state.model.final_bias,
                inputs: values,
            }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Prediction failed.");
        }

        setStatus(
            config.predictionStatusId,
            `Prediction = ${payload.prediction}, weighted sum z = ${payload.z.toFixed(4)} for input [${values.join(", ")}].`,
            false
        );
    } catch (error) {
        setStatus(config.predictionStatusId, error.message, true);
    }
}

function renderCalcWalkthrough(gate) {
    const config = getGateConfig(gate);
    const model = gateStates[gate].model;
    const tbody = element(config.walkthroughTableId).querySelector("tbody");

    tbody.innerHTML = model.walkthrough
        .map((row) => {
            const sampleText = `[${row.sample.join(", ")}] -> ${row.label}`;
            return `
                <tr>
                    <td>${escapeHtml(sampleText)}</td>
                    <td>${row.z.toFixed(4)}</td>
                    <td>${row.prediction}</td>
                    <td>${row.error}</td>
                    <td>${formatWeights(row.delta_weights)}</td>
                    <td>${row.delta_bias.toFixed(4)}</td>
                    <td>${formatWeights(row.new_weights)}</td>
                    <td>${row.new_bias.toFixed(4)}</td>
                </tr>
            `;
        })
        .join("");

    element(config.calcSummaryId).textContent =
        "Weighted sum: z = w1*x1 + ... + wn*xn + b | Step activation: y_hat = 1 if z >= 0 else 0 | " +
        "Update: wj = wj + alpha*(y - y_hat)*xj, b = b + alpha*(y - y_hat).";
}

function renderEpochTable(gate) {
    const config = getGateConfig(gate);
    const model = gateStates[gate].model;
    const tbody = element(config.epochTableId).querySelector("tbody");

    tbody.innerHTML = model.epoch_history
        .map((row) => {
            return `
                <tr>
                    <td>${row.epoch}</td>
                    <td>${row.total_error}</td>
                    <td>${formatWeights(row.weights)}</td>
                    <td>${row.bias.toFixed(4)}</td>
                </tr>
            `;
        })
        .join("");
}

function renderSummary(gate) {
    const config = getGateConfig(gate);
    const model = gateStates[gate].model;

    element(config.summaryId).textContent =
        `Final weights: ${formatWeights(model.final_weights)} | ` +
        `Final bias: ${model.final_bias.toFixed(4)} | ` +
        `Epochs: ${model.epochs_trained} | ` +
        `Converged: ${model.converged ? "Yes" : "No"}`;
}

function renderChart(gate) {
    const config = getGateConfig(gate);
    const model = gateStates[gate].model;
    const canvas = element(config.chartId);
    const ctx = canvas.getContext("2d");

    if (chartState[gate]) {
        chartState[gate].destroy();
    }

    chartState[gate] = new Chart(ctx, {
        type: "line",
        data: {
            labels: model.epoch_history.map((row) => row.epoch),
            datasets: [
                {
                    label: "Total Error",
                    data: model.epoch_history.map((row) => row.total_error),
                    borderColor: "rgba(5, 150, 105, 1)",
                    backgroundColor: "rgba(5, 150, 105, 0.12)",
                    fill: true,
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 },
                },
            },
        },
    });
}

function preparePredictionInputs(gate, featureCount) {
    const config = getGateConfig(gate);
    const container = element(config.predictionInputsId);
    container.innerHTML = "";

    for (let i = 0; i < featureCount; i += 1) {
        const block = document.createElement("div");
        block.className = "parameter-input";

        const label = document.createElement("label");
        label.setAttribute("for", `${gate}Pred${i}`);
        label.textContent = `x${i + 1}`;

        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.id = `${gate}Pred${i}`;
        input.placeholder = `Enter x${i + 1}`;

        block.appendChild(label);
        block.appendChild(input);
        container.appendChild(block);
    }
}

function parseWeightInput(raw) {
    const values = raw
        .split(",")
        .map((v) => v.trim())
        .filter((v) => v.length > 0)
        .map((v) => Number(v));

    if (!values.length || !values.every((v) => Number.isFinite(v))) {
        throw new Error("Initial weights must be a comma-separated numeric list.");
    }

    return values;
}

function getGateConfig(gate) {
    return {
        dataId: `${gate}Data`,
        lrId: `${gate}Lr`,
        epochsId: `${gate}Epochs`,
        weightsId: `${gate}Weights`,
        biasId: `${gate}Bias`,
        statusId: `${gate}Status`,
        calcSummaryId: `${gate}CalcSummary`,
        walkthroughTableId: `${gate}WalkthroughTable`,
        summaryId: `${gate}FinalSummary`,
        chartId: `${gate}Chart`,
        epochTableId: `${gate}EpochTable`,
        predictionInputsId: `${gate}PredictionInputs`,
        predictionStatusId: `${gate}PredictionStatus`,
    };
}

function formatWeights(weights) {
    return weights.map((w, idx) => `w${idx + 1}: ${Number(w).toFixed(4)}`).join(", ");
}

function setStatus(id, message, isError) {
    const status = element(id);
    status.textContent = message;
    status.classList.toggle("status-error", isError);
    status.classList.toggle("status-ok", !isError);
}

function element(id) {
    const el = document.getElementById(id);
    if (!el) {
        throw new Error(`Missing element: ${id}`);
    }
    return el;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
