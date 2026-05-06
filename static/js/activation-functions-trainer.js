const activationState = {
    activations: [],
    currentActivation: null,
    activationChart: null,
    derivativeChart: null,
};

const formulas = {
    binary_step: "a = 1 if z >= 0 else 0\nz = w*x + b",
    linear: "a = z\nz = w*x + b",
    sigmoid: "a = 1 / (1 + exp(-z))\nda/dz = a*(1-a)",
    tanh: "a = tanh(z)\nda/dz = 1 - a^2",
    relu: "a = max(0, z)\nda/dz = 1 for z>0 else 0",
    leaky_relu: "a = z if z>0 else alpha*z\nda/dz = 1 or alpha",
    elu: "a = z if z>0 else alpha*(exp(z)-1)\nda/dz = 1 or alpha*exp(z)",
    softplus: "a = log(1 + exp(z))\nda/dz = sigmoid(z)",
    swish: "a = z*sigmoid(z)\nda/dz = sigmoid(z) + z*sigmoid(z)*(1-sigmoid(z))",
    softmax: "a_i = exp(z_i) / sum_j exp(z_j)\n(visualized as class1 prob for [z1, z2=0])",
};

document.addEventListener("DOMContentLoaded", () => {
    wireModalControls();
    wireActions();
    loadActivationCards();
});

function wireModalControls() {
    document.querySelector(".modal-close").addEventListener("click", closeModal);
    document.getElementById("activationModal").addEventListener("click", (event) => {
        if (event.target.id === "activationModal") {
            closeModal();
        }
    });
}

function wireActions() {
    document.getElementById("analyzeBtn").addEventListener("click", () => {
        runAnalysis();
    });
}

function closeModal() {
    document.getElementById("activationModal").classList.remove("open");
    destroyCharts();
}

async function loadActivationCards() {
    try {
        const response = await fetch("/api/activation/functions");
        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Failed to load activation cards.");
        }

        activationState.activations = payload.activations || [];
        const grid = document.getElementById("activationCards");
        grid.innerHTML = "";

        activationState.activations.forEach((item) => {
            const card = document.createElement("article");
            card.className = "activation-card";
            card.innerHTML = `
                <div class="activation-top">
                    <div class="activation-name">${item.label}</div>
                    <span class="activation-level">${item.level}</span>
                </div>
                <div class="activation-meta">${item.category}</div>
                <div class="activation-best"><strong>Best use:</strong> ${item.best_use}</div>
                <button class="btn btn-secondary" type="button">Open Analyzer</button>
            `;
            card.addEventListener("click", () => openActivation(item.name));
            grid.appendChild(card);
        });
    } catch (error) {
        setStatus(error.message, true);
    }
}

function openActivation(name) {
    activationState.currentActivation = name;

    const meta = activationState.activations.find((item) => item.name === name);
    document.getElementById("modalTitle").textContent = `${meta?.label || name} Analyzer`;
    document.getElementById("modalSubtitle").textContent = `Interactive study for ${meta?.label || name}`;

    document.getElementById("resultsSection").style.display = "none";
    document.getElementById("analysisStatus").className = "status-message";
    document.getElementById("analysisStatus").textContent = "Configure values and click Analyze Activation.";
    document.getElementById("formulaCard").textContent = formulas[name] || "Formula preview unavailable.";

    document.getElementById("activationModal").classList.add("open");
    runAnalysis();
}

async function runAnalysis() {
    try {
        if (!activationState.currentActivation) {
            throw new Error("Select an activation card first.");
        }

        const weight = parseFloat(document.getElementById("weightInput").value);
        const bias = parseFloat(document.getElementById("biasInput").value);
        const alpha = parseFloat(document.getElementById("alphaInput").value);
        const xMin = parseFloat(document.getElementById("xminInput").value);
        const xMax = parseFloat(document.getElementById("xmaxInput").value);
        const sampleRaw = document.getElementById("sampleInput").value.trim();

        const inputValues = sampleRaw
            .split(",")
            .map((v) => Number(v.trim()))
            .filter((v) => Number.isFinite(v));

        if (!Number.isFinite(weight) || !Number.isFinite(bias) || !Number.isFinite(alpha)) {
            throw new Error("Weight, bias, and alpha must be valid numbers.");
        }
        if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMin >= xMax) {
            throw new Error("X range is invalid. Ensure xMin < xMax.");
        }

        setStatus("Analyzing activation behavior...", false);

        const response = await fetch("/api/activation/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                activation: activationState.currentActivation,
                weight,
                bias,
                alpha,
                x_min: xMin,
                x_max: xMax,
                points: 241,
                input_values: inputValues,
            }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Analysis failed.");
        }

        renderResults(payload);
        document.getElementById("resultsSection").style.display = "grid";
        setStatus("Analysis complete.", false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

function renderResults(model) {
    renderSummary(model);
    renderCharts(model);
    renderFormula(model.name);
    renderSampleTable(model.sample_table || []);
    renderImpactTable(model);
    renderInsights(model);
}

function renderSummary(model) {
    document.getElementById("resultsSummary").innerHTML = `
        <div class="summary-item"><div class="summary-label">Activation</div><div class="summary-value">${model.label}</div></div>
        <div class="summary-item"><div class="summary-label">Category</div><div class="summary-value">${model.category}</div></div>
        <div class="summary-item"><div class="summary-label">Weight / Bias</div><div class="summary-value">${Number(model.parameters.weight).toFixed(3)} / ${Number(model.parameters.bias).toFixed(3)}</div></div>
        <div class="summary-item"><div class="summary-label">Range</div><div class="summary-value">[${model.parameters.x_min}, ${model.parameters.x_max}]</div></div>
    `;
}

function renderCharts(model) {
    destroyCharts();

    const actCtx = document.getElementById("activationChart").getContext("2d");
    const derCtx = document.getElementById("derivativeChart").getContext("2d");

    activationState.activationChart = new Chart(actCtx, {
        type: "line",
        data: {
            labels: model.x_values,
            datasets: [
                {
                    label: `${model.label} output`,
                    data: model.y_values,
                    borderColor: "rgba(14, 116, 144, 1)",
                    backgroundColor: "rgba(14, 116, 144, 0.12)",
                    pointRadius: 0,
                    tension: 0.25,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { maxTicksLimit: 8 } },
            },
        },
    });

    activationState.derivativeChart = new Chart(derCtx, {
        type: "line",
        data: {
            labels: model.x_values,
            datasets: [
                {
                    label: `${model.label} derivative`,
                    data: model.derivative_values,
                    borderColor: "rgba(220, 38, 38, 1)",
                    backgroundColor: "rgba(220, 38, 38, 0.12)",
                    pointRadius: 0,
                    tension: 0.25,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { maxTicksLimit: 8 } },
            },
        },
    });
}

function renderFormula(name) {
    document.getElementById("formulaCard").textContent = formulas[name] || "Formula preview unavailable.";
}

function renderSampleTable(rows) {
    const tbody = document.querySelector("#sampleTable tbody");
    tbody.innerHTML = "";

    rows.forEach((row) => {
        tbody.innerHTML += `
            <tr>
                <td>${Number(row.input).toFixed(4)}</td>
                <td>${Number(row.pre_activation).toFixed(4)}</td>
                <td>${Number(row.activation).toFixed(6)}</td>
                <td>${Number(row.derivative).toFixed(6)}</td>
            </tr>
        `;
    });
}

function renderImpactTable(model) {
    const thead = document.querySelector("#impactTable thead");
    const tbody = document.querySelector("#impactTable tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (model.name === "softmax") {
        thead.innerHTML = "<tr><th>Feature</th><th>Class1 Probability</th><th>Class2 Probability</th></tr>";
        (model.impact_table || []).forEach((row) => {
            tbody.innerHTML += `
                <tr>
                    <td>${Number(row.feature).toFixed(3)}</td>
                    <td>${Number(row.class1_probability).toFixed(6)}</td>
                    <td>${Number(row.class2_probability).toFixed(6)}</td>
                </tr>
            `;
        });
    } else {
        thead.innerHTML = "<tr><th>Feature</th><th>Activated Output</th></tr>";
        (model.impact_table || []).forEach((row) => {
            tbody.innerHTML += `
                <tr>
                    <td>${Number(row.feature).toFixed(3)}</td>
                    <td>${Number(row.activated_output).toFixed(6)}</td>
                </tr>
            `;
        });
    }
}

function renderInsights(model) {
    const list = document.getElementById("insightList");
    list.innerHTML = "";

    const points = [
        `Best use: ${model.best_use}`,
        `Impact: ${model.impact}`,
        `Limitation: ${model.limitations}`,
    ];

    points.forEach((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        list.appendChild(li);
    });
}

function destroyCharts() {
    if (activationState.activationChart) {
        activationState.activationChart.destroy();
        activationState.activationChart = null;
    }
    if (activationState.derivativeChart) {
        activationState.derivativeChart.destroy();
        activationState.derivativeChart = null;
    }
}

function setStatus(message, isError) {
    const el = document.getElementById("analysisStatus");
    el.textContent = message;
    el.className = `status-message ${isError ? "error" : "success"}`;
}
