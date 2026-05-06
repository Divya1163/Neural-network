const backpropState = {
    currentTrainer: null,
    model: null,
    chartInstance: null,
};

const backpropConfig = {
    xor: {
        title: "XOR Backpropagation Trainer",
        subtitle: "Learn nonlinear learning with hidden neurons and gradient-based updates.",
    },
    custom: {
        title: "Custom Backpropagation Trainer",
        subtitle: "Train with your own CSV dataset and inspect convergence.",
    },
};

const defaultCustomDataset = [
    "x1,x2,label",
    "0,0,0",
    "0,1,1",
    "1,0,1",
    "1,1,0",
].join("\n");

document.addEventListener("DOMContentLoaded", () => {
    wireTrainerCards();
    wireModalControls();
    wireFormActions();
});

function wireTrainerCards() {
    document.querySelectorAll(".trainer-card").forEach((card) => {
        card.addEventListener("click", () => {
            openTrainerModal(card.dataset.trainer);
        });
    });
}

function wireModalControls() {
    document.querySelector(".modal-close").addEventListener("click", closeTrainerModal);
    document.getElementById("trainerModal").addEventListener("click", (event) => {
        if (event.target.id === "trainerModal") {
            closeTrainerModal();
        }
    });
}

function wireFormActions() {
    document.getElementById("loadSampleBtn").addEventListener("click", () => {
        loadSampleData(backpropState.currentTrainer);
    });

    document.getElementById("trainBtn").addEventListener("click", () => {
        trainBackprop();
    });

    document.getElementById("predictBtn").addEventListener("click", () => {
        predictBackprop();
    });
}

function openTrainerModal(trainer) {
    backpropState.currentTrainer = trainer;
    const modal = document.getElementById("trainerModal");
    const cfg = backpropConfig[trainer];

    document.getElementById("modalTitle").textContent = cfg.title;
    document.getElementById("modalSubtitle").textContent = cfg.subtitle;

    resetForm();
    loadSampleData(trainer);
    modal.classList.add("open");
}

function closeTrainerModal() {
    document.getElementById("trainerModal").classList.remove("open");
    if (backpropState.chartInstance) {
        backpropState.chartInstance.destroy();
        backpropState.chartInstance = null;
    }
}

function resetForm() {
    document.getElementById("dataInput").value = "";
    document.getElementById("learningRate").value = "0.5";
    document.getElementById("epochs").value = "10";
    document.getElementById("hiddenNeurons").value = "2";
    document.getElementById("trainingStatus").className = "status-message";
    document.getElementById("trainingStatus").textContent = "Waiting for training.";
    document.getElementById("resultsSection").style.display = "none";
    document.getElementById("predictionSection").style.display = "none";
    document.getElementById("predictionResult").style.display = "none";
    backpropState.model = null;
}

async function loadSampleData(trainer) {
    try {
        if (trainer === "custom") {
            document.getElementById("dataInput").value = defaultCustomDataset;
            setStatus("Loaded editable sample custom dataset.", false);
            return;
        }

        const response = await fetch(`/api/backprop/dataset/${trainer}`);
        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || `Failed to load ${trainer} dataset.`);
        }

        document.getElementById("dataInput").value = payload.dataset_csv;
        setStatus(`Loaded ${trainer.toUpperCase()} dataset.`, false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

async function trainBackprop() {
    try {
        const trainer = backpropState.currentTrainer;
        const datasetCsv = document.getElementById("dataInput").value.trim();
        const learningRate = parseFloat(document.getElementById("learningRate").value);
        const epochs = parseInt(document.getElementById("epochs").value, 10);
        const hiddenNeurons = parseInt(document.getElementById("hiddenNeurons").value, 10);

        if (!datasetCsv) {
            throw new Error("Training data CSV is required.");
        }
        if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
            throw new Error("Learning rate must be > 0 and <= 1.");
        }
        if (!Number.isInteger(epochs) || epochs < 1) {
            throw new Error("Epochs must be a positive integer.");
        }
        if (!Number.isInteger(hiddenNeurons) || hiddenNeurons < 1) {
            throw new Error("Hidden neurons must be at least 1.");
        }

        setStatus("Training model...", false);

        const response = await fetch("/api/backprop/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset: trainer,
                dataset_csv: datasetCsv,
                learning_rate: learningRate,
                epochs,
                hidden_neurons: hiddenNeurons,
            }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Training failed.");
        }

        backpropState.model = payload;
        renderResults(payload);
        preparePredictionInputs(payload.feature_count);

        document.getElementById("resultsSection").style.display = "block";
        document.getElementById("predictionSection").style.display = "block";
        setStatus(payload.converged ? "Training complete. Convergence reached." : "Training complete. Try more epochs for lower MSE.", false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

function renderResults(model) {
    renderSummary(model);
    renderWalkthrough(model.walkthrough || []);
    renderEpochTable(model.epoch_history || []);
    renderPredictionTable(model.predictions || []);
    renderChart(model.epoch_history || []);
}

function renderSummary(model) {
    const el = document.getElementById("resultsSummary");
    el.innerHTML = `
        <div class="summary-item"><div class="summary-label">Final MSE</div><div class="summary-value">${Number(model.final_mse).toFixed(6)}</div></div>
        <div class="summary-item"><div class="summary-label">Epochs</div><div class="summary-value">${model.epochs_trained}</div></div>
        <div class="summary-item"><div class="summary-label">Hidden Neurons</div><div class="summary-value">${model.hidden_neurons}</div></div>
        <div class="summary-item"><div class="summary-label">Converged</div><div class="summary-value">${model.converged ? "Yes" : "No"}</div></div>
    `;
}

function renderWalkthrough(rows) {
    const tbody = document.querySelector("#walkthroughTable tbody");
    tbody.innerHTML = rows.map((row) => {
        return `
            <tr>
                <td>[${row.sample.join(", ")}]</td>
                <td>${Number(row.output).toFixed(6)}</td>
                <td>${Number(row.error).toFixed(6)}</td>
                <td>${Number(row.delta_output).toFixed(6)}</td>
                <td>${row.delta_hidden.map((v) => Number(v).toFixed(6)).join(", ")}</td>
                <td>${row.delta_weights_hidden_output.map((v) => Number(v).toFixed(6)).join(", ")}</td>
                <td>${Number(row.delta_bias_output).toFixed(6)}</td>
            </tr>
        `;
    }).join("");
}

function renderEpochTable(rows) {
    const tbody = document.querySelector("#epochTable tbody");
    tbody.innerHTML = rows.map((row) => {
        return `
            <tr>
                <td>${row.epoch}</td>
                <td>${Number(row.mse).toFixed(6)}</td>
                <td>${Number(row.bias_output).toFixed(6)}</td>
                <td>${row.weights_hidden_output.map((v) => Number(v).toFixed(6)).join(", ")}</td>
            </tr>
        `;
    }).join("");
}

function renderPredictionTable(rows) {
    const tbody = document.querySelector("#predictionTable tbody");
    tbody.innerHTML = rows.map((row) => {
        return `
            <tr>
                <td>[${row.sample.join(", ")}]</td>
                <td>${row.target}</td>
                <td>${Number(row.probability).toFixed(6)}</td>
                <td>${row.class}</td>
            </tr>
        `;
    }).join("");
}

function renderChart(history) {
    const canvas = document.getElementById("convergenceChart");
    const ctx = canvas.getContext("2d");

    if (backpropState.chartInstance) {
        backpropState.chartInstance.destroy();
    }

    backpropState.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: history.map((row) => row.epoch),
            datasets: [{
                label: "MSE",
                data: history.map((row) => row.mse),
                borderColor: "rgba(37, 99, 235, 1)",
                backgroundColor: "rgba(37, 99, 235, 0.12)",
                fill: true,
                pointRadius: 3,
                tension: 0.2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true },
            },
        },
    });
}

function preparePredictionInputs(featureCount) {
    const wrap = document.getElementById("predictionInputs");
    wrap.innerHTML = "";

    for (let i = 0; i < featureCount; i += 1) {
        const block = document.createElement("div");
        const label = document.createElement("label");
        const input = document.createElement("input");

        label.textContent = `x${i + 1}`;
        label.setAttribute("for", `predInput${i}`);
        input.id = `predInput${i}`;
        input.type = "number";
        input.step = "any";
        input.placeholder = `Enter x${i + 1}`;

        block.appendChild(label);
        block.appendChild(input);
        wrap.appendChild(block);
    }
}

async function predictBackprop() {
    try {
        const model = backpropState.model;
        if (!model) {
            throw new Error("Train model first.");
        }

        const values = [];
        for (let i = 0; i < model.feature_count; i += 1) {
            const value = Number(document.getElementById(`predInput${i}`).value);
            if (!Number.isFinite(value)) {
                throw new Error(`Enter valid numeric value for x${i + 1}.`);
            }
            values.push(value);
        }

        const response = await fetch("/api/backprop/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                weights_input_hidden: model.final_params.weights_input_hidden,
                weights_hidden_output: model.final_params.weights_hidden_output,
                bias_hidden: model.final_params.bias_hidden,
                bias_output: model.final_params.bias_output,
                inputs: values,
            }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Prediction failed.");
        }

        const result = document.getElementById("predictionResult");
        const details = document.getElementById("predictionDetails");

        details.innerHTML = `
            <div><span class="pred-detail-label">Input:</span> [${payload.inputs.join(", ")}]</div>
            <div><span class="pred-detail-label">Probability:</span> ${Number(payload.probability).toFixed(6)}</div>
            <div><span class="pred-detail-label">Class:</span> ${payload.prediction}</div>
        `;

        result.style.display = "block";
    } catch (error) {
        setStatus(error.message, true);
    }
}

function setStatus(message, isError) {
    const el = document.getElementById("trainingStatus");
    el.textContent = message;
    el.className = `status-message ${isError ? "error" : "success"}`;
}
