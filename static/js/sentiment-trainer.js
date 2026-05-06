let currentModelId = null;

function formatPercent(v) {
    if (typeof v !== 'number') {
        return '-';
    }
    return `${(v * 100).toFixed(2)}%`;
}

function loadModelInfo() {
    fetch('/api/sentiment/model-info')
        .then((response) => response.json())
        .then((data) => {
            if (!data.success) {
                throw new Error(data.error || 'Unable to load model info');
            }

            currentModelId = data.model_id;

            document.getElementById('meta-model').textContent = data.model_name;
            document.getElementById('meta-notebook').textContent = data.source_notebook;
            document.getElementById('meta-artifact').textContent = `${data.artifact.format.toUpperCase()} - ${data.artifact.path}`;
            document.getElementById('meta-tokenizer').textContent = `num_words=${data.tokenizer.num_words}, max_len=${data.tokenizer.max_len}`;
            document.getElementById('meta-compile').textContent = `${data.compile.optimizer} / ${data.compile.loss}`;

            const metrics = data.runtime_metrics || {};
            const resultsContent = document.getElementById('results-content');
            resultsContent.innerHTML = `
                <div class="results-stats">
                    <div class="stat-box">
                        <div class="stat-label">Train Samples</div>
                        <div class="stat-value">${metrics.train_samples ?? '-'}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Test Samples</div>
                        <div class="stat-value">${metrics.test_samples ?? '-'}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Epochs</div>
                        <div class="stat-value">${metrics.epochs ?? '-'}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Test Accuracy</div>
                        <div class="stat-value">${formatPercent(metrics.test_accuracy)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Test Loss</div>
                        <div class="stat-value">${typeof metrics.test_loss === 'number' ? metrics.test_loss.toFixed(4) : '-'}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Classes</div>
                        <div class="stat-value">${(data.dataset.classes || []).join(', ')}</div>
                    </div>
                </div>
            `;

            const stepsContainer = document.getElementById('build-steps');
            stepsContainer.innerHTML = '';
            (data.build_steps || []).forEach((step, index) => {
                const item = document.createElement('div');
                item.className = 'step-line';
                item.innerHTML = `
                    <div class="step-index">${index + 1}</div>
                    <div class="step-text">${step}</div>
                `;
                stepsContainer.appendChild(item);
            });
        })
        .catch((error) => {
            const resultsContent = document.getElementById('results-content');
            resultsContent.innerHTML = `<div class="error-message">${error.message}</div>`;
        });
}

function makePrediction() {
    const text = document.getElementById('prediction-text').value.trim();
    const resultDiv = document.getElementById('prediction-result');
    const errorDiv = document.getElementById('prediction-error');

    // Clear previous results
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');

    if (!text) {
        errorDiv.textContent = 'Please enter text to classify.';
        errorDiv.classList.remove('hidden');
        return;
    }

    // Make prediction using cached notebook model (or auto-bootstrap server-side)
    fetch('/api/sentiment/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text,
            model_id: currentModelId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayPrediction(data);
        } else {
            errorDiv.textContent = 'Error: ' + (data.error || 'Unknown error');
            errorDiv.classList.remove('hidden');
        }
    })
    .catch(error => {
        errorDiv.textContent = 'Error: ' + error.message;
        errorDiv.classList.remove('hidden');
        console.error('Prediction error:', error);
    });
}

function displayPrediction(data) {
    const resultDiv = document.getElementById('prediction-result');
    const badge = document.getElementById('sentiment-badge');
    const confValue = document.getElementById('confidence-value');
    const confFill = document.getElementById('confidence-fill');
    const probDiv = document.getElementById('probabilities-breakdown');
    const cleanedText = document.getElementById('cleaned-text-display');

    // Update sentiment badge
    const sentiment = data.sentiment.toLowerCase();
    badge.className = `sentiment-badge ${sentiment}`;
    badge.textContent = `${sentiment.toUpperCase()} SENTIMENT`;

    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    confValue.textContent = confidence + '%';
    confFill.style.width = confidence + '%';

    // Update probabilities
    const probs = data.probabilities;
    probDiv.innerHTML = `
        <div style="font-weight: 600; color: #1e3a8a; margin-bottom: 12px; font-size: 0.95rem;">Sentiment Probabilities</div>
        ${Object.entries(probs).map(([label, prob]) => `
            <div class="prob-item">
                <span class="prob-label">${label.charAt(0).toUpperCase() + label.slice(1)}</span>
                <span class="prob-value">${(prob * 100).toFixed(2)}%</span>
            </div>
        `).join('')}
    `;

    // Update cleaned text
    cleanedText.textContent = data.cleaned_text || data.text;

    // Show result
    resultDiv.classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
});
