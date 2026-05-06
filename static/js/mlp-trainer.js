const mlpState = {
    currentTrainer: null,
    model: null,
    lossChart: null,
    accuracyChart: null,
    datasets: [],
};

const mlpConfig = {
    builtin: {
        title: "Built-in Dataset MLP Trainer",
        subtitle: "Load curated datasets and train a configurable MLP pipeline.",
    },
    custom: {
        title: "Custom Dataset MLP Trainer",
        subtitle: "Upload or paste your own CSV and run full MLP training.",
    },
};

document.addEventListener("DOMContentLoaded", () => {
    wireTrainerCards();
    wireModalControls();
    wireFormActions();
    fetchBuiltinDatasets();
});

function wireTrainerCards() {
    document.querySelectorAll(".trainer-card").forEach((card) => {
        card.addEventListener("click", () => openTrainerModal(card.dataset.trainer));
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
        loadSelectedBuiltIn();
    });

    document.getElementById("trainBtn").addEventListener("click", () => {
        trainMlp();
    });

    document.getElementById("datasetFile").addEventListener("change", onDatasetFileChange);
}

function openTrainerModal(trainer) {
    mlpState.currentTrainer = trainer;
    const cfg = mlpConfig[trainer] || mlpConfig.builtin;

    document.getElementById("modalTitle").textContent = cfg.title;
    document.getElementById("modalSubtitle").textContent = cfg.subtitle;

    resetForm();
    document.getElementById("trainerModal").classList.add("open");

    if (trainer === "builtin") {
        loadSelectedBuiltIn();
    } else {
        setStatus("Upload a CSV or paste custom data, then click Train MLP.", false);
    }
}

function closeTrainerModal() {
    document.getElementById("trainerModal").classList.remove("open");
    destroyCharts();
}

function resetForm() {
    document.getElementById("targetColumn").value = "";
    document.getElementById("featureColumns").value = "";
    document.getElementById("hiddenLayers").value = "";
    document.getElementById("learningRate").value = "0.01";
    document.getElementById("epochs").value = "250";
    document.getElementById("batchSize").value = "32";
    document.getElementById("testSize").value = "0.2";
    document.getElementById("randomState").value = "42";
    document.getElementById("rowThreshold").value = "1000";
    document.getElementById("datasetFile").value = "";
    document.getElementById("resultsSection").style.display = "none";
    document.getElementById("trainingStatus").className = "status-message";
    document.getElementById("trainingStatus").textContent = "Choose a mode and load data to start.";
    document.getElementById("resultsSummary").innerHTML = "";
    document.getElementById("architectureContainer").innerHTML = "";
    document.getElementById("pipelineSteps").innerHTML = "";
    document.querySelector("#stepLogTable tbody").innerHTML = "";
    document.querySelector("#predictionTable tbody").innerHTML = "";
    document.querySelector("#confusionTable thead").innerHTML = "";
    document.querySelector("#confusionTable tbody").innerHTML = "";
    mlpState.model = null;
    destroyCharts();
}

async function fetchBuiltinDatasets() {
    try {
        const response = await fetch("/api/mlp/datasets");
        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Failed to load built-in datasets.");
        }

        mlpState.datasets = payload.datasets || [];
        const select = document.getElementById("builtinDatasetSelect");
        select.innerHTML = "";

        mlpState.datasets.forEach((item) => {
            const option = document.createElement("option");
            option.value = item.name;
            option.textContent = `${item.label} (${item.source})`;
            select.appendChild(option);
        });

        if (mlpState.datasets.length > 0) {
            select.value = mlpState.datasets[0].name;
        }
    } catch (error) {
        setStatus(error.message, true);
    }
}

async function loadSelectedBuiltIn() {
    try {
        const dataset = document.getElementById("builtinDatasetSelect").value;
        if (!dataset) {
            throw new Error("Please select a built-in dataset first.");
        }

        const response = await fetch(`/api/mlp/dataset/${dataset}`);
        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Failed to load built-in dataset.");
        }

        document.getElementById("dataInput").value = payload.dataset_csv;
        setStatus(`Loaded '${dataset}' dataset CSV.`, false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

function onDatasetFileChange(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        document.getElementById("dataInput").value = String(reader.result || "").trim();
        setStatus(`Loaded file '${file.name}'.`, false);
    };
    reader.onerror = () => {
        setStatus("Could not read selected file.", true);
    };
    reader.readAsText(file);
}

async function trainMlp() {
    try {
        const datasetCsv = document.getElementById("dataInput").value.trim();
        if (!datasetCsv) {
            throw new Error("Dataset CSV is required.");
        }

        const learningRate = parseFloat(document.getElementById("learningRate").value);
        const epochs = parseInt(document.getElementById("epochs").value, 10);
        const batchSize = parseInt(document.getElementById("batchSize").value, 10);
        const testSize = parseFloat(document.getElementById("testSize").value);
        const randomState = parseInt(document.getElementById("randomState").value, 10);
        const rowThreshold = parseInt(document.getElementById("rowThreshold").value, 10);

        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new Error("Learning rate must be a positive number.");
        }
        if (!Number.isInteger(epochs) || epochs < 1) {
            throw new Error("Epochs must be a positive integer.");
        }
        if (!Number.isInteger(batchSize) || batchSize < 1) {
            throw new Error("Batch size must be a positive integer.");
        }

        const targetColumnRaw = document.getElementById("targetColumn").value.trim();
        const featureColumnsRaw = document.getElementById("featureColumns").value.trim();
        const hiddenLayersRaw = document.getElementById("hiddenLayers").value.trim();

        const selectedFeatureColumns = featureColumnsRaw
            ? featureColumnsRaw.split(",").map((item) => item.trim()).filter((item) => item.length > 0)
            : null;

        const hiddenLayersOverride = hiddenLayersRaw
            ? hiddenLayersRaw.split(",").map((item) => Number(item.trim())).filter((value) => Number.isInteger(value) && value > 0)
            : null;

        if (hiddenLayersRaw && (!hiddenLayersOverride || hiddenLayersOverride.length === 0)) {
            throw new Error("Hidden layers must be comma-separated positive integers, e.g. 32,16.");
        }

        const datasetName = mlpState.currentTrainer === "builtin" ? document.getElementById("builtinDatasetSelect").value : "custom";

        setStatus("Training MLP model...", false);

        const response = await fetch("/api/mlp/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset: datasetName,
                dataset_csv: datasetCsv,
                target_column: targetColumnRaw || null,
                selected_feature_columns: selectedFeatureColumns,
                row_threshold_for_drop: rowThreshold,
                test_size: testSize,
                random_state: randomState,
                learning_rate: learningRate,
                epochs,
                batch_size: batchSize,
                hidden_layers_override: hiddenLayersOverride,
            }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Training failed.");
        }

        mlpState.model = payload;
        renderResults(payload);
        document.getElementById("resultsSection").style.display = "grid";
        setStatus("Training completed successfully.", false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

function renderResults(model) {
    renderSummary(model);
    renderArchitecture(model.layer_dims || []);
    renderSteps(model.pipeline_steps || []);
    renderStepLog(model.step_log || []);
    renderConfusionMatrix(model.confusion_matrix || [], model.class_names || []);
    renderPredictionPreview(model.predictions_preview || []);
    renderCharts(model);
}

function renderSummary(model) {
    const summary = document.getElementById("resultsSummary");
    summary.innerHTML = `
        <div class="summary-item"><div class="summary-label">Final Test Accuracy</div><div class="summary-value">${(Number(model.final_test_accuracy) * 100).toFixed(2)}%</div></div>
        <div class="summary-item"><div class="summary-label">Final Test Loss</div><div class="summary-value">${Number(model.final_test_loss).toFixed(5)}</div></div>
        <div class="summary-item"><div class="summary-label">Train/Test Rows</div><div class="summary-value">${model.train_size} / ${model.test_size}</div></div>
        <div class="summary-item"><div class="summary-label">Layer Dimensions</div><div class="summary-value">${(model.layer_dims || []).join(" -> ")}</div></div>
    `;
}

function renderArchitecture(layerDims) {
    const container = document.getElementById("architectureContainer");
    if (!layerDims.length) {
        container.innerHTML = "<p>No architecture available.</p>";
        return;
    }

    const labels = layerDims.map((value, index) => {
        if (index === 0) {
            return `Input (${value})`;
        }
        if (index === layerDims.length - 1) {
            return `Output (${value})`;
        }
        return `Hidden ${index} (${value})`;
    });

    container.innerHTML = labels
        .map((label, index) => {
            const arrow = index < labels.length - 1 ? "<span class=\"arch-arrow\">→</span>" : "";
            return `<span class=\"arch-layer\"><span class=\"arch-badge\">${label}</span>${arrow}</span>`;
        })
        .join("");
}

function renderSteps(steps) {
    const list = document.getElementById("pipelineSteps");
    list.innerHTML = "";
    steps.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        list.appendChild(li);
    });
}

function renderStepLog(stepLog) {
    const tbody = document.querySelector("#stepLogTable tbody");
    tbody.innerHTML = "";

    stepLog.forEach((entry) => {
        const row = document.createElement("tr");
        const details = entry.summary
            ? entry.summary
            : `batch_size=${entry.batch_size}, train_loss=${Number(entry.train_loss).toFixed(5)}, grad_norms=[${(entry.gradient_norms || []).map((v) => Number(v).toFixed(4)).join(", ")}]`;

        row.innerHTML = `
            <td>${entry.epoch ?? "-"}</td>
            <td>${entry.batch ?? "-"}</td>
            <td>${details}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderConfusionMatrix(confusion, classNames) {
    const thead = document.querySelector("#confusionTable thead");
    const tbody = document.querySelector("#confusionTable tbody");

    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (!Array.isArray(confusion) || confusion.length === 0) {
        return;
    }

    const headerCells = ["True/Pred", ...classNames].map((label) => `<th>${label}</th>`).join("");
    thead.innerHTML = `<tr>${headerCells}</tr>`;

    confusion.forEach((row, index) => {
        const cells = [`<td>${classNames[index] ?? index}</td>`, ...row.map((value) => `<td>${value}</td>`)].join("");
        tbody.innerHTML += `<tr>${cells}</tr>`;
    });
}

function renderPredictionPreview(rows) {
    const tbody = document.querySelector("#predictionTable tbody");
    tbody.innerHTML = "";

    rows.forEach((row) => {
        tbody.innerHTML += `
            <tr>
                <td>${row.row_index}</td>
                <td>${row.true_class_name}</td>
                <td>${row.pred_class_name}</td>
                <td>${(Number(row.confidence) * 100).toFixed(2)}%</td>
            </tr>
        `;
    });
}

function renderCharts(model) {
    destroyCharts();

    const lossCtx = document.getElementById("lossChart").getContext("2d");
    const accCtx = document.getElementById("accuracyChart").getContext("2d");

    const labels = Array.from({ length: model.epochs_trained }, (_, index) => index + 1);

    mlpState.lossChart = new Chart(lossCtx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Train Loss",
                    data: model.train_loss_history,
                    borderColor: "rgba(14, 116, 144, 1)",
                    backgroundColor: "rgba(14, 116, 144, 0.15)",
                    fill: false,
                    tension: 0.25,
                    pointRadius: 2,
                },
                {
                    label: "Test Loss",
                    data: model.test_loss_history,
                    borderColor: "rgba(220, 38, 38, 1)",
                    backgroundColor: "rgba(220, 38, 38, 0.15)",
                    fill: false,
                    tension: 0.25,
                    pointRadius: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
        },
    });

    mlpState.accuracyChart = new Chart(accCtx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Train Accuracy",
                    data: model.train_acc_history,
                    borderColor: "rgba(5, 150, 105, 1)",
                    backgroundColor: "rgba(5, 150, 105, 0.15)",
                    fill: false,
                    tension: 0.25,
                    pointRadius: 2,
                },
                {
                    label: "Test Accuracy",
                    data: model.test_acc_history,
                    borderColor: "rgba(29, 78, 216, 1)",
                    backgroundColor: "rgba(29, 78, 216, 0.15)",
                    fill: false,
                    tension: 0.25,
                    pointRadius: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 1,
                },
            },
        },
    });
}

function destroyCharts() {
    if (mlpState.lossChart) {
        mlpState.lossChart.destroy();
        mlpState.lossChart = null;
    }
    if (mlpState.accuracyChart) {
        mlpState.accuracyChart.destroy();
        mlpState.accuracyChart = null;
    }
}

function setStatus(message, isError) {
    const el = document.getElementById("trainingStatus");
    el.textContent = message;
    el.className = `status-message ${isError ? "error" : "success"}`;
}
