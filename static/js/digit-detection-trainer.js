const digitState = {
    config: null,
    trained: false,
    selectedFileBase64: null,
    charts: {
        binaryCurve: null,
        multiCurve: null,
        binaryPerDigit: null,
        binaryOutputMean: null,
        predictMulti: null,
        predictBinary: null,
    },
};

document.addEventListener("DOMContentLoaded", () => {
    wireEvents();
    loadConfig();
    initPreviewCanvas();
});

function wireEvents() {
    document.getElementById("trainProjectBtn").addEventListener("click", trainProject);
    document.getElementById("digitImageInput").addEventListener("change", handleImageSelect);
    document.getElementById("predictBtn").addEventListener("click", runPrediction);
}

function initPreviewCanvas() {
    const canvas = document.getElementById("imagePreviewCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "14px Manrope";
    ctx.fillText("Image preview", 68, 112);
}

async function loadConfig() {
    try {
        const response = await fetch("/api/digit_project/config");
        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Could not load project config.");
        }

        digitState.config = payload;
        const activationSelect = document.getElementById("binaryActivation");
        activationSelect.innerHTML = "";
        (payload.binary_activations || []).forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            activationSelect.appendChild(option);
        });

        const d = payload.default || {};
        document.getElementById("binaryActivation").value = d.binary_activation || "sigmoid";
        document.getElementById("targetDigit").value = d.target_digit ?? 0;
        document.getElementById("trainLimit").value = d.train_limit ?? 9000;
        document.getElementById("testLimit").value = d.test_limit ?? 2000;
        document.getElementById("binaryEpochs").value = d.binary_epochs ?? 35;
        document.getElementById("multiclassEpochs").value = d.multiclass_epochs ?? 25;
        document.getElementById("learningRate").value = d.learning_rate ?? 0.08;
        document.getElementById("alphaLeaky").value = d.alpha_leaky ?? 0.01;

        setStatus("projectStatus", "Configuration loaded. Ready to train.", false);
    } catch (error) {
        setStatus("projectStatus", error.message, true);
    }
}

async function trainProject() {
    try {
        const payload = {
            binary_activation: document.getElementById("binaryActivation").value,
            target_digit: Number(document.getElementById("targetDigit").value),
            train_limit: Number(document.getElementById("trainLimit").value),
            test_limit: Number(document.getElementById("testLimit").value),
            binary_epochs: Number(document.getElementById("binaryEpochs").value),
            multiclass_epochs: Number(document.getElementById("multiclassEpochs").value),
            learning_rate: Number(document.getElementById("learningRate").value),
            alpha_leaky: Number(document.getElementById("alphaLeaky").value),
            random_state: Number(document.getElementById("randomState").value),
        };

        if (payload.target_digit < 0 || payload.target_digit > 9) {
            throw new Error("Target digit must be between 0 and 9.");
        }

        setStatus("projectStatus", "Training models on MNIST... this may take a minute on first run.", false);

        const response = await fetch("/api/digit_project/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || "Training failed.");
        }

        digitState.trained = true;
        renderTrainingResults(result);
        setStatus("projectStatus", "Training complete. You can now upload an image for prediction.", false);
        setStatus("predictStatus", "Model ready. Upload an image and click predict.", false);
    } catch (error) {
        setStatus("projectStatus", error.message, true);
    }
}

function renderTrainingResults(model) {
    document.getElementById("resultsSection").style.display = "grid";

    document.getElementById("resultsSummary").innerHTML = `
        <div class="summary-item"><div class="summary-label">Binary Activation</div><div class="summary-value">${model.binary_activation}</div></div>
        <div class="summary-item"><div class="summary-label">Binary Target Digit</div><div class="summary-value">${model.target_digit}</div></div>
        <div class="summary-item"><div class="summary-label">Binary Test Acc</div><div class="summary-value">${(model.binary_test_acc.at(-1) * 100).toFixed(2)}%</div></div>
        <div class="summary-item"><div class="summary-label">Multiclass Test Acc</div><div class="summary-value">${(model.multiclass_test_acc.at(-1) * 100).toFixed(2)}%</div></div>
    `;

    const labelsBinary = Array.from({ length: model.binary_train_loss.length }, (_, i) => i + 1);
    const labelsMulti = Array.from({ length: model.multiclass_train_loss.length }, (_, i) => i + 1);

    drawOrReplaceChart("binaryCurve", "binaryCurveChart", "line", labelsBinary, [
        { label: "Train Loss", data: model.binary_train_loss, borderColor: "#0ea5e9", pointRadius: 0, tension: 0.2 },
        { label: "Test Loss", data: model.binary_test_loss, borderColor: "#ef4444", pointRadius: 0, tension: 0.2 },
        { label: "Train Acc", data: model.binary_train_acc, borderColor: "#16a34a", pointRadius: 0, tension: 0.2, yAxisID: "y1" },
        { label: "Test Acc", data: model.binary_test_acc, borderColor: "#1d4ed8", pointRadius: 0, tension: 0.2, yAxisID: "y1" },
    ], true);

    drawOrReplaceChart("multiCurve", "multiCurveChart", "line", labelsMulti, [
        { label: "Train Loss", data: model.multiclass_train_loss, borderColor: "#0ea5e9", pointRadius: 0, tension: 0.2 },
        { label: "Test Loss", data: model.multiclass_test_loss, borderColor: "#ef4444", pointRadius: 0, tension: 0.2 },
        { label: "Train Acc", data: model.multiclass_train_acc, borderColor: "#16a34a", pointRadius: 0, tension: 0.2, yAxisID: "y1" },
        { label: "Test Acc", data: model.multiclass_test_acc, borderColor: "#1d4ed8", pointRadius: 0, tension: 0.2, yAxisID: "y1" },
    ], true);

    const digitLabels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

    drawOrReplaceChart("binaryPerDigit", "binaryPerDigitChart", "bar", digitLabels, [
        { label: "Binary Accuracy", data: model.per_digit_binary_accuracy, backgroundColor: "rgba(14,116,144,0.7)", borderColor: "#0e7490", borderWidth: 1 },
    ], false);

    drawOrReplaceChart("binaryOutputMean", "binaryOutputMeanChart", "bar", digitLabels, [
        { label: "Mean Output", data: model.per_digit_binary_output_mean, backgroundColor: "rgba(29,78,216,0.65)", borderColor: "#1d4ed8", borderWidth: 1 },
    ], false);

    renderConfusion(model.multiclass_confusion || []);
}

function renderConfusion(matrix) {
    const thead = document.querySelector("#confusionTable thead");
    const tbody = document.querySelector("#confusionTable tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (!Array.isArray(matrix) || matrix.length !== 10) {
        return;
    }

    const header = ["True\\Pred", ...Array.from({ length: 10 }, (_, i) => i)].map((v) => `<th>${v}</th>`).join("");
    thead.innerHTML = `<tr>${header}</tr>`;

    matrix.forEach((row, idx) => {
        const cells = [`<td>${idx}</td>`, ...row.map((v) => `<td>${v}</td>`)].join("");
        tbody.innerHTML += `<tr>${cells}</tr>`;
    });
}

function handleImageSelect(event) {
    const file = event.target.files?.[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        const base64 = String(reader.result || "");
        digitState.selectedFileBase64 = base64;
        drawPreview(base64);
    };
    reader.onerror = () => setStatus("predictStatus", "Could not read selected image.", true);
    reader.readAsDataURL(file);
}

function drawPreview(base64) {
    const canvas = document.getElementById("imagePreviewCanvas");
    const ctx = canvas.getContext("2d");
    const image = new Image();

    image.onload = () => {
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const ratio = Math.min(canvas.width / image.width, canvas.height / image.height);
        const drawW = image.width * ratio;
        const drawH = image.height * ratio;
        const x = (canvas.width - drawW) / 2;
        const y = (canvas.height - drawH) / 2;

        ctx.drawImage(image, x, y, drawW, drawH);
    };
    image.src = base64;
}

async function runPrediction() {
    try {
        if (!digitState.trained) {
            throw new Error("Train the model before prediction.");
        }
        if (!digitState.selectedFileBase64) {
            throw new Error("Upload an image first.");
        }

        setStatus("predictStatus", "Running prediction...", false);

        const response = await fetch("/api/digit_project/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: digitState.selectedFileBase64 }),
        });

        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || "Prediction failed.");
        }

        document.getElementById("predictResults").style.display = "grid";
        document.getElementById("predictSummary").innerHTML = `
            <div class="summary-item"><div class="summary-label">Predicted Digit</div><div class="summary-value">${result.predicted_digit_multiclass}</div></div>
            <div class="summary-item"><div class="summary-label">Binary Target</div><div class="summary-value">${result.binary_target_digit}</div></div>
            <div class="summary-item"><div class="summary-label">Binary Output</div><div class="summary-value">${result.binary_output.toFixed(6)}</div></div>
            <div class="summary-item"><div class="summary-label">Binary Decision</div><div class="summary-value">${result.binary_statement}</div></div>
        `;

        const labels = Array.from({ length: 10 }, (_, i) => `${i}`);

        drawOrReplaceChart("predictMulti", "predictMulticlassChart", "bar", labels, [
            {
                label: "Softmax Probability",
                data: result.multiclass_probabilities,
                backgroundColor: "rgba(16,185,129,0.65)",
                borderColor: "#10b981",
                borderWidth: 1,
            },
        ], false);

        drawOrReplaceChart("predictBinary", "predictBinaryChart", "bar", labels, [
            {
                label: "Per-digit score",
                data: result.per_digit_output,
                backgroundColor: "rgba(59,130,246,0.65)",
                borderColor: "#3b82f6",
                borderWidth: 1,
            },
        ], false);

        setStatus("predictStatus", "Prediction complete.", false);
    } catch (error) {
        setStatus("predictStatus", error.message, true);
    }
}

function drawOrReplaceChart(stateKey, canvasId, type, labels, datasets, dualAxis) {
    if (digitState.charts[stateKey]) {
        digitState.charts[stateKey].destroy();
    }

    const ctx = document.getElementById(canvasId).getContext("2d");
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { ticks: { maxTicksLimit: 12 } },
        },
    };

    if (dualAxis) {
        options.scales.y = { beginAtZero: true };
        options.scales.y1 = {
            beginAtZero: true,
            min: 0,
            max: 1,
            position: "right",
            grid: { drawOnChartArea: false },
        };
    } else {
        options.scales.y = { beginAtZero: true };
    }

    digitState.charts[stateKey] = new Chart(ctx, {
        type,
        data: { labels, datasets },
        options,
    });
}

function setStatus(elementId, message, isError) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `status-message ${isError ? "error" : "success"}`;
}
