// Perceptron Trainer Modal - Main JS

const trainerState = {
    currentTrainer: null,
    model: null,
    chartInstance: null,
};

const trainerConfig = {
    and: {
        title: "AND Gate Trainer",
        subtitle: "Train a model to learn the AND logic operation",
    },
    or: {
        title: "OR Gate Trainer",
        subtitle: "Train a model to learn the OR logic operation",
    },
    custom: {
        title: "Custom Perceptron Analyzer",
        subtitle: "Train a perceptron on your own dataset",
    },
};

document.addEventListener("DOMContentLoaded", () => {
    wireTrainerCards();
    wireModalControls();
    wireFormActions();
});

// ============ TRAINER CARD INTERACTIONS ============
function wireTrainerCards() {
    document.querySelectorAll(".trainer-card").forEach((card) => {
        card.addEventListener("click", () => {
            const trainer = card.dataset.trainer;
            openTrainerModal(trainer);
        });
    });
}

function openTrainerModal(trainer) {
    trainerState.currentTrainer = trainer;
    const config = trainerConfig[trainer];
    const modal = document.getElementById("trainerModal");

    document.getElementById("modalTitle").textContent = config.title;
    document.getElementById("modalSubtitle").textContent = config.subtitle;

    // Reset form
    resetForm();

    // Load sample data
    loadSampleData(trainer);

    // Show modal
    modal.classList.add("open");
}

function closeTrainerModal() {
    document.getElementById("trainerModal").classList.remove("open");
    if (trainerState.chartInstance) {
        trainerState.chartInstance.destroy();
        trainerState.chartInstance = null;
    }
}

// ============ MODAL CONTROLS ============
function wireModalControls() {
    document.querySelector(".modal-close").addEventListener("click", closeTrainerModal);

    document.getElementById("trainerModal").addEventListener("click", (e) => {
        if (e.target.id === "trainerModal") {
            closeTrainerModal();
        }
    });
}

// ============ FORM ACTIONS ============
function wireFormActions() {
    document.getElementById("loadSampleBtn").addEventListener("click", () => {
        loadSampleData(trainerState.currentTrainer);
    });

    document.getElementById("trainBtn").addEventListener("click", () => {
        trainModel();
    });

    document.getElementById("predictBtn").addEventListener("click", () => {
        makePrediiction();
    });
}

function resetForm() {
    document.getElementById("dataInput").value = "";
    document.getElementById("learningRate").value = "1";
    document.getElementById("epochs").value = "10";
    document.getElementById("initialWeights").value = "0,0";
    document.getElementById("initialBias").value = "0";
    document.getElementById("trainingStatus").textContent = "";
    document.getElementById("trainingStatus").className = "status-message";
    document.getElementById("resultsSection").style.display = "none";
    document.getElementById("predictionSection").style.display = "none";
    document.getElementById("predictionResult").style.display = "none";
    trainerState.model = null;
}

// ============ LOAD SAMPLE DATA ============
async function loadSampleData(trainer) {
    const statusEl = document.getElementById("trainingStatus");
    try {
        const response = await fetch(`/api/perceptron/dataset/${trainer}`);
        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || `Could not load ${trainer} dataset.`);
        }

        document.getElementById("dataInput").value = payload.dataset_csv;
        setStatus("Loading sample data successful.", false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

// ============ TRAIN MODEL ============
async function trainModel() {
    const trainer = trainerState.currentTrainer;
    const statusEl = document.getElementById("trainingStatus");

    try {
        // Validate inputs
        const learningRate = parseFloat(document.getElementById("learningRate").value);
        const epochs = parseInt(document.getElementById("epochs").value, 10);
        const initialBias = parseFloat(document.getElementById("initialBias").value);
        const datasetCsv = document.getElementById("dataInput").value.trim();
        const weightsRaw = document.getElementById("initialWeights").value;

        if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
            throw new Error("Learning rate must be > 0 and <= 1.");
        }
        if (!Number.isInteger(epochs) || epochs < 1) {
            throw new Error("Epochs must be a positive integer.");
        }
        if (!Number.isFinite(initialBias)) {
            throw new Error("Initial bias must be a valid number.");
        }
        if (!datasetCsv) {
            throw new Error("Training data is required.");
        }

        const initialWeights = parseWeights(weightsRaw);

        // Call API
        setStatus("Training in progress...", false);

        const response = await fetch("/api/perceptron/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                gate: trainer,
                dataset_csv: datasetCsv,
                learning_rate: learningRate,
                epochs,
                initial_weights: initialWeights,
                initial_bias: initialBias,
            }),
        });

        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Training failed.");
        }

        trainerState.model = payload;

        // Show results
        renderResults();
        setStatus("Training completed successfully!", false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

// ============ RENDER RESULTS ============
function renderResults() {
    const model = trainerState.model;

    // Show sections
    document.getElementById("resultsSection").style.display = "block";
    document.getElementById("predictionSection").style.display = "block";

    // Summary
    const summaryHtml = `
        <div class="summary-item">
            <div class="summary-label">Final Weights</div>
            <div class="summary-value">${formatWeights(model.final_weights)}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Final Bias</div>
            <div class="summary-value">${model.final_bias.toFixed(4)}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Epochs Trained</div>
            <div class="summary-value">${model.epochs_trained}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Converged</div>
            <div class="summary-value">${model.converged ? "✓ Yes" : "✗ No"}</div>
        </div>
    `;
    document.querySelector(".results-summary").innerHTML = summaryHtml;

    // Walkthrough table
    const walkthroughBody = document.querySelector("#walkthroughTable tbody");
    walkthroughBody.innerHTML = model.walkthrough
        .map((row) => {
            const sampleText = `[${row.sample.join(", ")}]`;
            return `
                <tr>
                    <td>${sampleText}</td>
                    <td>${row.z.toFixed(4)}</td>
                    <td>${row.prediction}</td>
                    <td>${row.error}</td>
                    <td>${formatWeights([row.delta_weights[0]])}</td>
                    <td>${row.delta_bias.toFixed(4)}</td>
                    <td>${formatWeights(row.new_weights)}</td>
                    <td>${row.new_bias.toFixed(4)}</td>
                </tr>
            `;
        })
        .join("");

    // Chart
    renderChart();

    // Epoch table
    const epochBody = document.querySelector("#epochTable tbody");
    epochBody.innerHTML = model.epoch_history
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

    // Setup prediction inputs
    setupPredictionInputs(model.feature_count);
}

function renderChart() {
    const model = trainerState.model;
    const canvas = document.getElementById("convergenceChart");
    const ctx = canvas.getContext("2d");

    if (trainerState.chartInstance) {
        trainerState.chartInstance.destroy();
    }

    trainerState.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: model.epoch_history.map((row) => row.epoch),
            datasets: [
                {
                    label: "Total Error",
                    data: model.epoch_history.map((row) => row.total_error),
                    borderColor: "rgba(5, 150, 105, 1)",
                    backgroundColor: "rgba(5, 150, 105, 0.1)",
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: "rgba(5, 150, 105, 1)",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: "top",
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 },
                },
            },
        },
    });
}

// ============ PREDICTION ============
function setupPredictionInputs(featureCount) {
    const container = document.getElementById("predictionInputs");
    container.innerHTML = "";

    for (let i = 0; i < featureCount; i++) {
        const div = document.createElement("div");
        const label = document.createElement("label");
        label.textContent = `x${i + 1}`;
        label.setAttribute("for", `pred${i}`);

        const input = document.createElement("input");
        input.type = "number";
        input.id = `pred${i}`;
        input.step = "any";
        input.placeholder = `Enter x${i + 1}`;

        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    }
}

async function makePrediiction() {
    const model = trainerState.model;

    if (!model) {
        setStatus("Train the model first.", true);
        return;
    }

    try {
        const values = [];
        for (let i = 0; i < model.feature_count; i++) {
            const num = Number(document.getElementById(`pred${i}`).value);
            if (!Number.isFinite(num)) {
                throw new Error(`Enter valid numeric value for x${i + 1}.`);
            }
            values.push(num);
        }

        const response = await fetch("/api/perceptron/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                weights: model.final_weights,
                bias: model.final_bias,
                inputs: values,
            }),
        });

        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Prediction failed.");
        }

        // Show result
        const resultDiv = document.getElementById("predictionResult");
        const detailsDiv = document.getElementById("predictionDetails");

        detailsDiv.innerHTML = `
            <div class="pred-detail">
                <span class="pred-detail-label">Input:</span> [${values.join(", ")}]
            </div>
            <div class="pred-detail">
                <span class="pred-detail-label">Weighted Sum (z):</span> ${payload.z.toFixed(4)}
            </div>
            <div class="pred-detail">
                <span class="pred-detail-label">Prediction:</span> <strong style="font-size: 1.2em; color: var(--primary-green);">${payload.prediction}</strong>
            </div>
        `;

        resultDiv.style.display = "block";
    } catch (error) {
        alert(error.message);
    }
}

// ============ UTILITIES ============
function parseWeights(raw) {
    const values = raw
        .split(",")
        .map((v) => v.trim())
        .filter((v) => v.length > 0)
        .map((v) => Number(v));

    if (!values.length || !values.every((v) => Number.isFinite(v))) {
        throw new Error("Initial weights must be comma-separated numbers.");
    }

    return values;
}

function formatWeights(weights) {
    return weights.map((w, idx) => `w${idx + 1}: ${Number(w).toFixed(4)}`).join(", ");
}

function setStatus(message, isError) {
    const el = document.getElementById("trainingStatus");
    el.textContent = message;
    el.className = `status-message ${isError ? "error" : "success"}`;
}
