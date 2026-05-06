(function () {
    const uploadInput = document.getElementById('det-upload');
    const confidenceInput = document.getElementById('det-confidence');
    const confidenceLabel = document.getElementById('det-confidence-label');
    const personOnlyInput = document.getElementById('person-only');
    const smoothModeInput = document.getElementById('smooth-mode');
    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');

    const webcamInput = document.getElementById('webcam-input');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const imageInputPreview = document.getElementById('image-input-preview');
    const detectionOutput = document.getElementById('detection-output');
    const detectionSummary = document.getElementById('detection-summary');
    const objectsCountEl = document.getElementById('objects-count');
    const personsCountEl = document.getElementById('persons-count');
    const latencyMsEl = document.getElementById('latency-ms');
    const captureCanvas = document.getElementById('frame-capture-canvas');

    let webcamStream = null;
    let webcamRafId = null;
    let webcamRunning = false;
    let inFlight = false;
    let nextAllowedAt = 0;

    const FAST_INTERVAL = 140;
    const NORMAL_INTERVAL = 240;

    function getConfidence() {
        return parseFloat(confidenceInput.value || '0.4');
    }

    function isPersonOnly() {
        return Boolean(personOnlyInput && personOnlyInput.checked);
    }

    function isSmoothMode() {
        return !smoothModeInput || smoothModeInput.checked;
    }

    function setSummary(text) {
        detectionSummary.textContent = text;
    }

    function formatSummary(payload) {
        const count = payload.object_count || 0;
        const people = payload.person_count || 0;
        const model = payload.model || 'YOLOv8';
        const mode = payload.person_only ? ' (person-only)' : '';
        return `${model}${mode}: ${count} object(s), ${people} person(s).`;
    }

    function updateMetrics(payload, latencyMs) {
        if (objectsCountEl) {
            objectsCountEl.textContent = String(payload.object_count || 0);
        }
        if (personsCountEl) {
            personsCountEl.textContent = String(payload.person_count || 0);
        }
        if (latencyMsEl) {
            latencyMsEl.textContent = `${Math.round(latencyMs)} ms`;
        }
    }

    async function detectImageBase64(payload, endpoint) {
        const body = endpoint.indexOf('webcam') >= 0
            ? { frame: payload, confidence: getConfidence(), max_dim: 640, person_only: isPersonOnly() }
            : { image: payload, confidence: getConfidence(), max_dim: 960, person_only: isPersonOnly() };

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || 'Detection failed');
        }
        return data;
    }

    function drawOverlay(detections, sourceShape) {
        const width = webcamInput.videoWidth;
        const height = webcamInput.videoHeight;
        if (!width || !height || !overlayCanvas) {
            return;
        }

        overlayCanvas.width = width;
        overlayCanvas.height = height;

        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        ctx.lineWidth = 2;
        ctx.font = '14px Segoe UI';

        const sourceH = Array.isArray(sourceShape) && sourceShape.length === 2 ? Number(sourceShape[0]) : height;
        const sourceW = Array.isArray(sourceShape) && sourceShape.length === 2 ? Number(sourceShape[1]) : width;
        const scaleX = sourceW > 0 ? width / sourceW : 1;
        const scaleY = sourceH > 0 ? height / sourceH : 1;

        (detections || []).forEach((det) => {
            const box = det.box || [];
            if (box.length !== 4) {
                return;
            }
            const x1 = box[0] * scaleX;
            const y1 = box[1] * scaleY;
            const x2 = box[2] * scaleX;
            const y2 = box[3] * scaleY;
            const isPerson = det.label === 'person';
            ctx.strokeStyle = isPerson ? '#ef4444' : '#22c55e';
            ctx.fillStyle = ctx.strokeStyle;
            ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
            const label = `${det.label} ${Number(det.confidence || 0).toFixed(2)}`;
            ctx.fillRect(x1, Math.max(0, y1 - 18), ctx.measureText(label).width + 10, 18);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 5, Math.max(12, y1 - 5));
        });
    }

    uploadInput.addEventListener('change', async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) {
            return;
        }

        stopWebcam();

        const reader = new FileReader();
        reader.onload = async function () {
            const imgBase64 = String(reader.result || '');
            if (!imgBase64) {
                return;
            }

            imageInputPreview.src = imgBase64;
            imageInputPreview.style.display = 'block';
            webcamInput.style.display = 'none';
            detectionOutput.style.display = 'block';

            setSummary('Running detection on uploaded image...');
            try {
                const t0 = performance.now();
                const payload = await detectImageBase64(imgBase64, '/api/opencv/detect-image');
                detectionOutput.src = payload.annotated_image;
                setSummary(formatSummary(payload));
                updateMetrics(payload, performance.now() - t0);
                drawOverlay([], null);
            } catch (error) {
                setSummary(`Error: ${error.message}`);
            }
        };

        reader.readAsDataURL(file);
    });

    confidenceInput.addEventListener('input', () => {
        confidenceLabel.textContent = Number(getConfidence()).toFixed(2);
    });

    async function processWebcamFrame() {
        if (!webcamRunning || inFlight) {
            return;
        }

        const now = performance.now();
        if (now < nextAllowedAt) {
            return;
        }

        const width = webcamInput.videoWidth;
        const height = webcamInput.videoHeight;
        if (!width || !height) {
            return;
        }

        inFlight = true;
        const targetMax = isSmoothMode() ? 640 : 800;
        const scale = Math.min(1, targetMax / Math.max(width, height));
        captureCanvas.width = Math.max(1, Math.round(width * scale));
        captureCanvas.height = Math.max(1, Math.round(height * scale));
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(webcamInput, 0, 0, captureCanvas.width, captureCanvas.height);

        try {
            const quality = isSmoothMode() ? 0.65 : 0.82;
            const frame = captureCanvas.toDataURL('image/jpeg', quality);
            const t0 = performance.now();
            const payload = await detectImageBase64(frame, '/api/opencv/detect-webcam-frame');
            drawOverlay(payload.detections || [], payload.image_shape || null);
            setSummary(formatSummary(payload));
            const latency = performance.now() - t0;
            updateMetrics(payload, latency);
            const baseGap = isSmoothMode() ? FAST_INTERVAL : NORMAL_INTERVAL;
            nextAllowedAt = performance.now() + Math.max(baseGap, latency * 0.45);
        } catch (error) {
            setSummary(`Error: ${error.message}`);
            nextAllowedAt = performance.now() + 450;
        } finally {
            inFlight = false;
        }
    }

    function webcamLoop() {
        if (!webcamRunning) {
            return;
        }
        processWebcamFrame();
        webcamRafId = window.requestAnimationFrame(webcamLoop);
    }

    async function startWebcam() {
        if (webcamRunning) {
            return;
        }

        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 960 },
                    height: { ideal: 540 },
                    frameRate: { ideal: 24, max: 30 },
                },
                audio: false,
            });
            webcamInput.srcObject = webcamStream;
            webcamInput.style.display = 'block';
            imageInputPreview.style.display = 'none';
            detectionOutput.style.display = 'none';
            webcamRunning = true;
            nextAllowedAt = performance.now();
            drawOverlay([], null);
            setSummary('Webcam started. Running live detection...');
            webcamLoop();
        } catch (error) {
            setSummary(`Unable to start webcam: ${error.message}`);
        }
    }

    function stopWebcam() {
        webcamRunning = false;
        inFlight = false;

        if (webcamRafId) {
            window.cancelAnimationFrame(webcamRafId);
            webcamRafId = null;
        }

        if (webcamStream) {
            webcamStream.getTracks().forEach((track) => track.stop());
            webcamStream = null;
        }

        webcamInput.srcObject = null;
        drawOverlay([], null);
    }

    startWebcamBtn.addEventListener('click', startWebcam);
    stopWebcamBtn.addEventListener('click', () => {
        stopWebcam();
        detectionOutput.style.display = detectionOutput.src ? 'block' : 'none';
        setSummary('Webcam stopped. Upload an image or restart webcam.');
    });

    window.addEventListener('beforeunload', stopWebcam);
})();
