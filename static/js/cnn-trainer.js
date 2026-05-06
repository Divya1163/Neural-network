document.addEventListener('DOMContentLoaded', function () {
    // Modal handling
    const modals = document.querySelectorAll('.modal');
    const modalButtons = document.querySelectorAll('[data-modal-target]');
    const closeButtons = document.querySelectorAll('.close-button');

    modalButtons.forEach(button => {
        button.addEventListener('click', () => {
            const selector = button.dataset.modalTarget;
            const modal = document.querySelector(selector) || document.getElementById(selector);
            openModal(modal);
        });
    });

    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const modal = button.closest('.modal');
            closeModal(modal);
        });
    });

    window.addEventListener('click', (event) => {
        modals.forEach(modal => {
            if (event.target == modal) {
                closeModal(modal);
            }
        });
    });

    function openModal(modal) {
        if (modal == null) return;
        modal.style.display = 'block';
    }

    function closeModal(modal) {
        if (modal == null) return;
        modal.style.display = 'none';
    }

    // Convolution Demo
    initializeConvolutionDemo();
    initializeUnderstandingImagesLab();
    initializeArchitecturesStudio();

    // CNN Builder
    const generateBtn = document.getElementById('builder-generate');
    if (generateBtn) {
        generateBtn.addEventListener('click', generateArchitecture);
        generateArchitecture(); // Initial generation
    }
});

function initializeConvolutionDemo() {
    const uploadInput = document.getElementById('conv-upload');
    const sizeToggle = document.getElementById('size-toggle');
    const layerSelect = document.getElementById('layer-select');
    const nextStepBtn = document.getElementById('next-step');
    const animateBtn = document.getElementById('animate-conv');
    const pauseBtn = document.getElementById('pause-conv');
    const whyOpenBtn = document.getElementById('why-open');
    const whyCloseBtn = document.getElementById('why-close');
    const whyPopup = document.getElementById('why-popup');

    const rawShape = document.getElementById('raw-shape');
    const normShape = document.getElementById('norm-shape');
    const rawPreview = document.getElementById('raw-tensor-preview');
    const normPreview = document.getElementById('norm-tensor-preview');
    const equationLine = document.getElementById('equation-line');
    const equationValue = document.getElementById('equation-value');
    const pixelInspector = document.getElementById('pixel-inspector');

    const originalCanvas = document.getElementById('original-canvas');
    const featureCanvas = document.getElementById('feature-canvas');
    const heatmapCanvas = document.getElementById('heatmap-canvas');
    const originalCtx = originalCanvas.getContext('2d');
    const featureCtx = featureCanvas.getContext('2d');
    const heatmapCtx = heatmapCanvas.getContext('2d');

    const dlOriginal = document.getElementById('download-original');
    const dlFeature = document.getElementById('download-feature');
    const dlHeatmap = document.getElementById('download-heatmap');
    const exampleA = document.getElementById('example-cat');
    const exampleB = document.getElementById('example-shape');

    if (!uploadInput || !sizeToggle || !layerSelect || !originalCanvas || !featureCanvas || !heatmapCanvas) {
        return;
    }

    const layerOrder = ['edge', 'blur', 'sharpen', 'emboss'];
    const kernels = {
        edge: [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        blur: [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ],
        sharpen: [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        emboss: [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ]
    };

    const state = {
        size: Number(sizeToggle.value),
        originalData: [],
        normalizedData: [],
        outputData: [],
        path: [],
        stepIndex: 0,
        timer: null,
        isRunning: false
    };

    function initArrays() {
        const count = state.size * state.size;
        state.originalData = new Array(count).fill(0);
        state.normalizedData = new Array(count).fill(0);
        state.outputData = new Array(count).fill(0);
    }

    function resizeCanvases() {
        [originalCanvas, featureCanvas, heatmapCanvas].forEach(canvas => {
            canvas.width = state.size;
            canvas.height = state.size;
        });
    }

    function buildPath() {
        const stride = Math.max(1, Math.floor(state.size / 20));
        const positions = [];
        for (let y = 1; y < state.size - 1; y += stride) {
            for (let x = 1; x < state.size - 1; x += stride) {
                positions.push({ x, y });
            }
        }
        if (positions.length === 0) {
            positions.push({ x: 1, y: 1 });
        }
        state.path = positions;
        state.stepIndex = 0;
    }

    function drawFromArray(ctx, arr, mode) {
        const img = ctx.createImageData(state.size, state.size);
        for (let i = 0; i < arr.length; i++) {
            let v = arr[i];
            if (mode === 'normalized') {
                v = Math.min(1, Math.max(0, (v + 1) / 2));
            } else {
                v = Math.min(1, Math.max(0, v));
            }
            const channel = Math.round(v * 255);
            const idx = i * 4;
            img.data[idx] = channel;
            img.data[idx + 1] = channel;
            img.data[idx + 2] = channel;
            img.data[idx + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
    }

    function drawHeatmap(arr) {
        const img = heatmapCtx.createImageData(state.size, state.size);
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < arr.length; i++) {
            min = Math.min(min, arr[i]);
            max = Math.max(max, arr[i]);
        }
        const range = Math.max(1e-6, max - min);
        for (let i = 0; i < arr.length; i++) {
            const t = (arr[i] - min) / range;
            const r = Math.round(255 * t);
            const g = Math.round(100 * (1 - t));
            const b = Math.round(255 * (1 - t));
            const idx = i * 4;
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255;
        }
        heatmapCtx.putImageData(img, 0, 0);
    }

    function annotateKernel(x, y) {
        originalCtx.save();
        originalCtx.strokeStyle = '#2563eb';
        originalCtx.lineWidth = Math.max(1, Math.round(state.size / 100));
        originalCtx.strokeRect(x - 1, y - 1, 3, 3);
        originalCtx.restore();
    }

    function normalizeData() {
        state.normalizedData = state.originalData.map(v => (v - 0.5) / 0.5);
    }

    function previewTensor(container, arr, label) {
        const rows = Math.min(4, state.size);
        const cols = Math.min(6, state.size);
        const lines = [];
        for (let y = 0; y < rows; y++) {
            const rowVals = [];
            for (let x = 0; x < cols; x++) {
                rowVals.push(arr[y * state.size + x].toFixed(3));
            }
            lines.push('[' + rowVals.join(', ') + ']');
        }
        container.textContent = label + '\n' + lines.join('\n') + '\n...';
    }

    function refreshAllViews() {
        normalizeData();
        drawFromArray(originalCtx, state.originalData, 'raw');
        drawFromArray(featureCtx, state.outputData, 'raw');
        drawHeatmap(state.outputData);
        rawShape.textContent = `Shape: ${state.size} x ${state.size} x 1`;
        normShape.textContent = `Shape: ${state.size} x ${state.size} x 1 (mean=0.5, std=0.5)`;
        previewTensor(rawPreview, state.originalData, 'Raw tensor sample:');
        previewTensor(normPreview, state.normalizedData, 'Normalized tensor sample:');
        equationLine.textContent = 'output = sum(input_patch * kernel)';
        equationValue.textContent = 'Ready. Click Play Animation to see per-step math.';
    }

    function computeAt(x, y) {
        const kernel = kernels[layerSelect.value];
        let sum = 0;
        const parts = [];
        for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
                const px = x + kx;
                const py = y + ky;
                const inputVal = state.normalizedData[py * state.size + px];
                const kv = kernel[ky + 1][kx + 1];
                sum += inputVal * kv;
                parts.push(`${inputVal.toFixed(2)}*${kv.toFixed(2)}`);
            }
        }
        return { sum, expression: parts.join(' + ') };
    }

    function animationStep() {
        if (state.stepIndex >= state.path.length) {
            state.isRunning = false;
            clearInterval(state.timer);
            state.timer = null;
            animateBtn.textContent = 'Replay Animation';
            return;
        }

        drawFromArray(originalCtx, state.originalData, 'raw');
        const point = state.path[state.stepIndex];
        const result = computeAt(point.x, point.y);
        state.outputData[point.y * state.size + point.x] = Math.max(0, Math.min(1, (result.sum + 1) / 2));

        drawFromArray(featureCtx, state.outputData, 'raw');
        drawHeatmap(state.outputData);
        annotateKernel(point.x, point.y);

        equationLine.textContent = `output(${point.x}, ${point.y}) = ${result.expression}`;
        equationValue.textContent = `Result = ${result.sum.toFixed(4)} | Activated map value = ${state.outputData[point.y * state.size + point.x].toFixed(4)}`;

        state.stepIndex += 1;
    }

    function startAnimation() {
        if (state.timer) {
            clearInterval(state.timer);
        }
        if (state.stepIndex >= state.path.length) {
            state.stepIndex = 0;
            state.outputData.fill(0);
        }
        state.isRunning = true;
        animateBtn.textContent = 'Playing...';
        const delay = state.size > 100 ? 60 : 150;
        state.timer = setInterval(animationStep, delay);
    }

    function pauseAnimation() {
        if (state.timer) {
            clearInterval(state.timer);
            state.timer = null;
        }
        state.isRunning = false;
        animateBtn.textContent = 'Resume Animation';
    }

    function setExample(name) {
        for (let y = 0; y < state.size; y++) {
            for (let x = 0; x < state.size; x++) {
                const idx = y * state.size + x;
                if (name === 'shape') {
                    const cx = state.size / 2;
                    const cy = state.size / 2;
                    const dist = Math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    state.originalData[idx] = dist < state.size * 0.28 ? 0.95 : 0.08;
                } else {
                    state.originalData[idx] = (x / state.size) * 0.7 + (y % 20 < 10 ? 0.25 : 0.05);
                }
            }
        }
        state.outputData.fill(0);
        buildPath();
        refreshAllViews();
    }

    function loadImageToTensor(image) {
        const offscreen = document.createElement('canvas');
        offscreen.width = state.size;
        offscreen.height = state.size;
        const offCtx = offscreen.getContext('2d');
        offCtx.drawImage(image, 0, 0, state.size, state.size);
        const data = offCtx.getImageData(0, 0, state.size, state.size).data;
        for (let i = 0; i < state.size * state.size; i++) {
            const idx = i * 4;
            const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / (3 * 255);
            state.originalData[i] = gray;
        }
        state.outputData.fill(0);
        buildPath();
        refreshAllViews();
    }

    function downloadCanvas(canvas, filename) {
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = filename;
        link.click();
    }

    function setupHover(canvas, dataRef, sourceName) {
        canvas.addEventListener('mousemove', event => {
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor(((event.clientX - rect.left) / rect.width) * state.size);
            const y = Math.floor(((event.clientY - rect.top) / rect.height) * state.size);
            if (x < 0 || y < 0 || x >= state.size || y >= state.size) {
                return;
            }
            const value = dataRef[y * state.size + x] || 0;
            pixelInspector.textContent = `${sourceName} pixel (${x}, ${y}) = ${value.toFixed(4)}`;
        });
    }

    setupHover(originalCanvas, state.originalData, 'Original');
    setupHover(featureCanvas, state.outputData, 'Feature');
    setupHover(heatmapCanvas, state.outputData, 'Heatmap base');

    uploadInput.addEventListener('change', event => {
        const file = event.target.files && event.target.files[0];
        if (!file) {
            return;
        }
        const reader = new FileReader();
        reader.onload = e => {
            const img = new Image();
            img.onload = () => loadImageToTensor(img);
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });

    sizeToggle.addEventListener('change', () => {
        pauseAnimation();
        state.size = Number(sizeToggle.value);
        resizeCanvases();
        initArrays();
        buildPath();
        setExample('cat');
    });

    layerSelect.addEventListener('change', () => {
        state.outputData.fill(0);
        state.stepIndex = 0;
        refreshAllViews();
    });

    nextStepBtn.addEventListener('click', () => {
        const current = layerOrder.indexOf(layerSelect.value);
        const next = (current + 1) % layerOrder.length;
        layerSelect.value = layerOrder[next];
        layerSelect.dispatchEvent(new Event('change'));
    });

    animateBtn.addEventListener('click', () => {
        if (state.isRunning) {
            pauseAnimation();
            return;
        }
        startAnimation();
    });

    pauseBtn.addEventListener('click', pauseAnimation);

    whyOpenBtn.addEventListener('click', () => {
        whyPopup.classList.add('open');
        whyPopup.setAttribute('aria-hidden', 'false');
    });
    whyCloseBtn.addEventListener('click', () => {
        whyPopup.classList.remove('open');
        whyPopup.setAttribute('aria-hidden', 'true');
    });
    whyPopup.addEventListener('click', event => {
        if (event.target === whyPopup) {
            whyPopup.classList.remove('open');
            whyPopup.setAttribute('aria-hidden', 'true');
        }
    });

    exampleA.addEventListener('click', () => setExample('cat'));
    exampleB.addEventListener('click', () => setExample('shape'));

    dlOriginal.addEventListener('click', () => downloadCanvas(originalCanvas, `original-${state.size}.png`));
    dlFeature.addEventListener('click', () => downloadCanvas(featureCanvas, `feature-map-${state.size}.png`));
    dlHeatmap.addEventListener('click', () => downloadCanvas(heatmapCanvas, `heatmap-${state.size}.png`));

    resizeCanvases();
    initArrays();
    buildPath();
    setExample('cat');
}

function generateArchitecture() {
    const inputSize = parseInt(document.getElementById('builder-input').value, 10);
    const conv1 = parseInt(document.getElementById('builder-conv1').value, 10);
    const conv2 = parseInt(document.getElementById('builder-conv2').value, 10);
    const dense = parseInt(document.getElementById('builder-dense').value, 10);

    const pool1 = Math.floor(inputSize / 2);
    const pool2 = Math.floor(pool1 / 2);
    const flatten = pool2 * pool2 * conv2;

    // Update text
    const textDiv = document.getElementById('architecture-text');
    textDiv.innerHTML = `
        <p><strong>CNN Architecture Summary</strong></p>
        <ul class="arch-list">
            <li><strong>Input:</strong> ${inputSize} × ${inputSize} × 1</li>
            <li><strong>Conv1 + ReLU:</strong> ${inputSize} × ${inputSize} × ${conv1}</li>
            <li><strong>MaxPool(2×2):</strong> ${pool1} × ${pool1} × ${conv1}</li>
            <li><strong>Conv2 + ReLU:</strong> ${pool1} × ${pool1} × ${conv2}</li>
            <li><strong>MaxPool(2×2):</strong> ${pool2} × ${pool2} × ${conv2}</li>
            <li><strong>Flatten:</strong> ${flatten} units</li>
            <li><strong>Dense:</strong> ${dense} units</li>
            <li><strong>Output:</strong> 10 classes</li>
        </ul>
        <p><em>Total parameters: ~${estimateParameters(conv1, conv2, dense, flatten).toLocaleString()}</em></p>
    `;

    // Generate visual diagram
    const diagramDiv = document.getElementById('architecture-diagram');
    diagramDiv.innerHTML = createArchitectureDiagram(inputSize, conv1, conv2, dense, pool1, pool2);
}

function initializeUnderstandingImagesLab() {
    const upload = document.getElementById('imglab-upload');
    const sizeSelect = document.getElementById('imglab-size');
    const preprocess = document.getElementById('imglab-preprocess');
    const kernelSelect = document.getElementById('imglab-kernel');
    const strideSlider = document.getElementById('imglab-stride');
    const paddingSlider = document.getElementById('imglab-padding');
    const filtersSlider = document.getElementById('imglab-filters');
    const strideValue = document.getElementById('imglab-stride-value');
    const paddingValue = document.getElementById('imglab-padding-value');
    const filtersValue = document.getElementById('imglab-filters-value');
    const runBtn = document.getElementById('imglab-run');
    const playBtn = document.getElementById('imglab-play');
    const pauseBtn = document.getElementById('imglab-pause');
    const nextBtn = document.getElementById('imglab-next');
    const customWrap = document.getElementById('imglab-custom-kernel');
    const kernelGrid = document.getElementById('imglab-kernel-grid');

    const rawShape = document.getElementById('imglab-raw-shape');
    const preShape = document.getElementById('imglab-pre-shape');
    const rawPreview = document.getElementById('imglab-raw-preview');
    const prePreview = document.getElementById('imglab-pre-preview');
    const equation = document.getElementById('imglab-equation');
    const equationSum = document.getElementById('imglab-equation-sum');
    const payoff = document.getElementById('imglab-payoff');
    const liveContext = document.getElementById('imglab-live-context');
    const resultGuide = document.getElementById('imglab-result-guide');
    const useCase = document.getElementById('imglab-use-case');
    const inspector = document.getElementById('imglab-inspector');
    const gallery = document.getElementById('imglab-gallery');

    const presetEdges = document.getElementById('preset-edges');
    const presetSmooth = document.getElementById('preset-smooth');
    const presetSharpen = document.getElementById('preset-sharpen');
    const impactList = document.getElementById('imglab-impact-list');

    const chatThread = document.getElementById('imglab-chat-thread');
    const chatChoices = document.getElementById('imglab-chat-choices');
    const chatStart = document.getElementById('imglab-chat-start');
    const chatReset = document.getElementById('imglab-chat-reset');

    const originalCanvas = document.getElementById('imglab-original');
    const preCanvas = document.getElementById('imglab-preprocessed');
    const featureCanvas = document.getElementById('imglab-feature');
    const heatmapCanvas = document.getElementById('imglab-heatmap');

    if (!upload || !runBtn || !originalCanvas || !featureCanvas) {
        return;
    }

    const state = {
        imageDataUrl: '',
        serverData: null,
        trace: [],
        traceIndex: 0,
        timer: null,
        liveFeature: null,
        running: false,
        guideStep: 0,
    };

    function updateSliderLabels() {
        strideValue.textContent = strideSlider.value;
        paddingValue.textContent = paddingSlider.value;
        filtersValue.textContent = filtersSlider.value;
        updateLiveContext();
    }

    function updateLiveContext() {
        const stride = Number(strideSlider.value);
        const padding = Number(paddingSlider.value);
        const filters = Number(filtersSlider.value);
        const kernel = kernelSelect.value;

        const strideMsg = stride === 1
            ? 'Stride 1 keeps maximum detail.'
            : `Stride ${stride} scans faster but skips fine details.`;
        const padMsg = padding === 0
            ? 'Padding 0 trims borders and can lose edge information.'
            : `Padding ${padding} preserves border context and larger outputs.`;
        const filterMsg = filters === 1
            ? '1 filter means one pattern detector.'
            : `${filters} filters means multiple pattern detectors.`;

        if (liveContext) {
            liveContext.textContent = `Current setup: ${kernel} kernel. ${strideMsg} ${padMsg} ${filterMsg}`;
        }

        if (impactList) {
            const preprocessText = {
                normalize: 'Normalize centers values around zero for stable feature responses.',
                minmax: 'Min-Max scales values to [0,1] for comparable intensity ranges.',
                invert: 'Invert flips dark and bright regions to test contrast sensitivity.',
                threshold: 'Threshold keeps hard boundaries and removes shades.'
            };
            impactList.innerHTML = `
                <li><strong>Resize:</strong> ${sizeSelect.value} x ${sizeSelect.value} controls detail level and compute cost.</li>
                <li><strong>Kernel:</strong> ${kernelSelect.value} defines what visual pattern is amplified.</li>
                <li><strong>Stride:</strong> ${strideMsg}</li>
                <li><strong>Padding:</strong> ${padMsg}</li>
                <li><strong>Filters:</strong> ${filterMsg}</li>
                <li><strong>Preprocessing:</strong> ${preprocessText[preprocess.value] || preprocessText.normalize}</li>
            `;
        }
    }

    function appendChatMessage(role, text, typing = false) {
        if (!chatThread) {
            return;
        }
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        chatThread.appendChild(bubble);

        if (!typing || role !== 'bot') {
            bubble.textContent = text;
            chatThread.scrollTop = chatThread.scrollHeight;
            return;
        }

        let i = 0;
        const timer = setInterval(() => {
            bubble.textContent = text.slice(0, i);
            i += 1;
            chatThread.scrollTop = chatThread.scrollHeight;
            if (i > text.length) {
                clearInterval(timer);
            }
        }, 12);
    }

    function setChatChoices(choices) {
        if (!chatChoices) {
            return;
        }
        chatChoices.innerHTML = '';
        choices.forEach(choice => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'choice-chip';
            btn.textContent = choice.label;
            btn.addEventListener('click', choice.onClick);
            chatChoices.appendChild(btn);
        });
    }

    function askGuideStep(step) {
        state.guideStep = step;
        if (step === 0) {
            appendChatMessage('bot', 'Hi. I will guide you like a CNN tutor. First, upload an image so we can convert it into a tensor.', true);
            setChatChoices([
                {
                    label: 'Upload image now',
                    onClick: () => {
                        appendChatMessage('user', 'Upload image now');
                        upload.click();
                    }
                },
                {
                    label: 'Use 28 x 28 learning mode',
                    onClick: () => {
                        appendChatMessage('user', 'Use 28 x 28 learning mode');
                        sizeSelect.value = '28';
                        updateLiveContext();
                        appendChatMessage('bot', 'Great for beginners. Smaller image means easier to inspect each patch.', true);
                    }
                }
            ]);
            return;
        }

        if (step === 1) {
            appendChatMessage('bot', 'What do you want to detect first?', true);
            setChatChoices([
                { label: 'Boundaries (Edge detection)', onClick: () => { appendChatMessage('user', 'Boundaries (Edge detection)'); applyPreset('edges'); askGuideStep(2); } },
                { label: 'Smooth noisy image (Blur)', onClick: () => { appendChatMessage('user', 'Smooth noisy image (Blur)'); applyPreset('smooth'); askGuideStep(2); } },
                { label: 'Highlight details (Sharpen)', onClick: () => { appendChatMessage('user', 'Highlight details (Sharpen)'); applyPreset('sharpen'); askGuideStep(2); } }
            ]);
            return;
        }

        if (step === 2) {
            appendChatMessage('bot', 'Choose stride. Stride 1 is best for learning because it scans every location.', true);
            setChatChoices([
                { label: 'Stride 1 (best detail)', onClick: () => { appendChatMessage('user', 'Stride 1 (best detail)'); strideSlider.value = '1'; updateSliderLabels(); askGuideStep(3); } },
                { label: 'Stride 2 (faster, less detail)', onClick: () => { appendChatMessage('user', 'Stride 2 (faster, less detail)'); strideSlider.value = '2'; updateSliderLabels(); askGuideStep(3); } }
            ]);
            return;
        }

        if (step === 3) {
            appendChatMessage('bot', 'Choose padding. Padding 1 keeps border information.', true);
            setChatChoices([
                { label: 'Padding 1 (keep borders)', onClick: () => { appendChatMessage('user', 'Padding 1 (keep borders)'); paddingSlider.value = '1'; updateSliderLabels(); askGuideStep(4); } },
                { label: 'Padding 0 (valid only)', onClick: () => { appendChatMessage('user', 'Padding 0 (valid only)'); paddingSlider.value = '0'; updateSliderLabels(); askGuideStep(4); } }
            ]);
            return;
        }

        if (step === 4) {
            appendChatMessage('bot', 'How many filters? More filters detect more pattern types.', true);
            setChatChoices([
                { label: '1 filter (simple)', onClick: () => { appendChatMessage('user', '1 filter (simple)'); filtersSlider.value = '1'; updateSliderLabels(); askGuideStep(5); } },
                { label: '3 filters (balanced)', onClick: () => { appendChatMessage('user', '3 filters (balanced)'); filtersSlider.value = '3'; updateSliderLabels(); askGuideStep(5); } },
                { label: '5 filters (richer features)', onClick: () => { appendChatMessage('user', '5 filters (richer features)'); filtersSlider.value = '5'; updateSliderLabels(); askGuideStep(5); } }
            ]);
            return;
        }

        if (step === 5) {
            appendChatMessage('bot', 'Choose preprocessing. I recommend Normalize for stable convolution responses.', true);
            setChatChoices([
                { label: 'Normalize (recommended)', onClick: () => { appendChatMessage('user', 'Normalize (recommended)'); preprocess.value = 'normalize'; updateLiveContext(); askGuideStep(6); } },
                { label: 'Min-Max', onClick: () => { appendChatMessage('user', 'Min-Max'); preprocess.value = 'minmax'; updateLiveContext(); askGuideStep(6); } },
                { label: 'Threshold', onClick: () => { appendChatMessage('user', 'Threshold'); preprocess.value = 'threshold'; updateLiveContext(); askGuideStep(6); } }
            ]);
            return;
        }

        appendChatMessage('bot', 'Perfect. Your setup is ready. Click Generate Output to see how each choice affects the feature map.', true);
        setChatChoices([
            {
                label: 'Generate Output',
                onClick: () => {
                    appendChatMessage('user', 'Generate Output');
                    runLab();
                }
            }
        ]);
    }

    function applyPreset(name) {
        if (name === 'edges') {
            kernelSelect.value = 'edge detection';
            preprocess.value = 'normalize';
            strideSlider.value = '1';
            paddingSlider.value = '1';
            filtersSlider.value = '2';
            if (useCase) {
                useCase.textContent = 'Use case: Edge maps are used in early CNN layers for stroke boundaries, lane lines, and shape outlines.';
            }
        } else if (name === 'smooth') {
            kernelSelect.value = 'blur';
            preprocess.value = 'minmax';
            strideSlider.value = '1';
            paddingSlider.value = '1';
            filtersSlider.value = '1';
            if (useCase) {
                useCase.textContent = 'Use case: Blur can suppress noise before feature extraction in low-quality images.';
            }
        } else if (name === 'sharpen') {
            kernelSelect.value = 'sharpen';
            preprocess.value = 'normalize';
            strideSlider.value = '1';
            paddingSlider.value = '1';
            filtersSlider.value = '2';
            if (useCase) {
                useCase.textContent = 'Use case: Sharpen emphasizes local contrast and can make subtle details easier to detect.';
            }
        }
        customWrap.classList.add('hidden');
        updateSliderLabels();
        updateLiveContext();
    }

    function imageToCanvas(canvas, src) {
        if (!src) return;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
        };
        img.src = src;
    }

    function matrixPreview(matrix, limitRows, limitCols) {
        if (!Array.isArray(matrix) || matrix.length === 0) return 'No data';
        const rows = Math.min(limitRows, matrix.length);
        const cols = Math.min(limitCols, matrix[0].length);
        const lines = [];
        for (let r = 0; r < rows; r++) {
            const vals = [];
            for (let c = 0; c < cols; c++) {
                vals.push(Number(matrix[r][c]).toFixed(3));
            }
            lines.push('[' + vals.join(', ') + ']');
        }
        return lines.join('\n') + '\n...';
    }

    function drawLiveFeature() {
        if (!state.liveFeature || !state.serverData) return;
        const h = state.serverData.output_shape[0];
        const w = state.serverData.output_shape[1];
        const ctx = featureCanvas.getContext('2d');
        const img = ctx.createImageData(w, h);

        let min = Infinity;
        let max = -Infinity;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const v = state.liveFeature[y][x];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        const range = Math.max(1e-9, max - min);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const v = (state.liveFeature[y][x] - min) / range;
                const idx = (y * w + x) * 4;
                const g = Math.round(v * 255);
                img.data[idx] = g;
                img.data[idx + 1] = g;
                img.data[idx + 2] = g;
                img.data[idx + 3] = 255;
            }
        }
        featureCanvas.width = w;
        featureCanvas.height = h;
        ctx.putImageData(img, 0, 0);
    }

    function drawKernelHighlight(step) {
        if (!state.serverData) return;
        imageToCanvas(originalCanvas, state.serverData.raw_image);
        const ctx = originalCanvas.getContext('2d');
        const size = state.serverData.input_shape[0];
        const scaleX = originalCanvas.width / size;
        const scaleY = originalCanvas.height / size;
        ctx.save();
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = Math.max(1, Math.round(size / 120));
        const px = Math.max(0, step.patch_x);
        const py = Math.max(0, step.patch_y);
        ctx.strokeRect(px * scaleX, py * scaleY, 3 * scaleX, 3 * scaleY);
        ctx.restore();
    }

    function resetLiveFeature() {
        if (!state.serverData) return;
        const h = state.serverData.output_shape[0];
        const w = state.serverData.output_shape[1];
        state.liveFeature = Array.from({ length: h }, () => Array.from({ length: w }, () => 0));
        drawLiveFeature();
    }

    function animationStep() {
        if (!state.trace || state.traceIndex >= state.trace.length) {
            state.running = false;
            if (state.timer) {
                clearInterval(state.timer);
                state.timer = null;
            }
            playBtn.textContent = 'Replay';
            return;
        }

        const step = state.trace[state.traceIndex];
        const oy = step.out_y;
        const ox = step.out_x;
        if (state.liveFeature && state.liveFeature[oy] && state.liveFeature[oy][ox] !== undefined) {
            state.liveFeature[oy][ox] = step.sum;
        }

        drawLiveFeature();
        drawKernelHighlight(step);
        equation.textContent = `output(${ox}, ${oy}) = ${step.equation}`;
        equationSum.textContent = `sum = ${Number(step.sum).toFixed(5)}`;
        state.traceIndex += 1;
    }

    function pauseAnimation() {
        if (state.timer) {
            clearInterval(state.timer);
            state.timer = null;
        }
        state.running = false;
    }

    function playAnimation() {
        if (!state.serverData || !state.trace.length) {
            return;
        }
        if (state.traceIndex >= state.trace.length) {
            state.traceIndex = 0;
            resetLiveFeature();
        }
        pauseAnimation();
        state.running = true;
        playBtn.textContent = 'Playing...';
        state.timer = setInterval(animationStep, 120);
    }

    function readCustomKernel() {
        if (!kernelGrid) return null;
        const values = Array.from(kernelGrid.querySelectorAll('input')).map(inp => Number(inp.value || 0));
        if (values.length !== 9) return null;
        return [
            values.slice(0, 3),
            values.slice(3, 6),
            values.slice(6, 9)
        ];
    }

    async function runLab() {
        if (!state.imageDataUrl) {
            equationSum.textContent = 'Upload an image first.';
            appendChatMessage('bot', 'I still need an image before we can run convolution. Please upload one now.', true);
            setChatChoices([
                {
                    label: 'Upload image',
                    onClick: () => {
                        appendChatMessage('user', 'Upload image');
                        upload.click();
                    }
                }
            ]);
            return;
        }

        pauseAnimation();
        equationSum.textContent = 'Running Python convolution backend...';
        runBtn.disabled = true;

        try {
            const payload = {
                image: state.imageDataUrl,
                kernel_mode: kernelSelect.value,
                custom_kernel: kernelSelect.value === 'custom' ? readCustomKernel() : null,
                stride: Number(strideSlider.value),
                padding: Number(paddingSlider.value),
                num_filters: Number(filtersSlider.value),
                resize_to: Number(sizeSelect.value),
                preprocess_type: preprocess.value,
            };

            const response = await fetch('/api/cnn/image-conv-lab', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Failed to run convolution lab');
            }

            state.serverData = data;
            state.trace = data.trace || [];
            state.traceIndex = 0;

            rawShape.textContent = `Raw tensor shape: ${data.input_shape.join(' x ')}`;
            preShape.textContent = `Output shape: ${data.output_shape.join(' x ')} | stride=${data.stride}, padding=${data.padding}`;
            rawPreview.textContent = matrixPreview(data.raw_matrix, 5, 6);
            prePreview.textContent = matrixPreview(data.preprocessed_matrix, 5, 6);
            equation.textContent = 'output(i, j) = sum(input_patch * kernel)';
            equationSum.textContent = 'Press Play to animate the red kernel sweep.';
            if (payoff && data.explain) {
                payoff.textContent = `Learning payoff: ${data.explain.kernel_focus} ${data.explain.stride_padding} ${data.explain.texture}`;
            }

            if (resultGuide && data.explain) {
                resultGuide.innerHTML = `
                    <li><strong>Original:</strong> raw pixel grid (${data.input_shape.join(' x ')}).</li>
                    <li><strong>Preprocessed:</strong> ${data.explain.preprocess_note}</li>
                    <li><strong>Feature map:</strong> ${data.explain.kernel_focus}</li>
                    <li><strong>Heatmap:</strong> brighter regions indicate stronger activation from this kernel.</li>
                `;
            }

            if (useCase && data.explain) {
                useCase.textContent = `Use case: ${data.explain.use_case}`;
            }

            imageToCanvas(originalCanvas, data.raw_image);
            imageToCanvas(preCanvas, data.preprocessed_image);
            imageToCanvas(heatmapCanvas, data.heatmap_image);
            resetLiveFeature();

            gallery.innerHTML = '';
            (data.feature_maps || []).forEach(fm => {
                const card = document.createElement('div');
                card.className = 'feature-card';
                card.innerHTML = `
                    <img src="${fm.image}" alt="${fm.name}">
                    <div class="title">${fm.name}</div>
                    <p class="meta">shape: ${fm.shape[0]} x ${fm.shape[1]}</p>
                    <p class="meta">min: ${Number(fm.min).toFixed(3)} | max: ${Number(fm.max).toFixed(3)}</p>
                `;
                gallery.appendChild(card);
            });
        } catch (err) {
            equationSum.textContent = `Error: ${err.message}`;
        } finally {
            runBtn.disabled = false;
        }
    }

    function setupHover(canvas, sourceName, valueGetter) {
        canvas.addEventListener('mousemove', event => {
            if (!state.serverData) return;
            const rect = canvas.getBoundingClientRect();
            const relX = (event.clientX - rect.left) / rect.width;
            const relY = (event.clientY - rect.top) / rect.height;
            const info = valueGetter(relX, relY);
            if (!info) return;
            inspector.textContent = `${sourceName} (${info.x}, ${info.y}) -> ${Number(info.value).toFixed(6)}`;
        });
    }

    setupHover(originalCanvas, 'Original', (rx, ry) => {
        if (!state.serverData || !state.serverData.raw_matrix) return null;
        const h = state.serverData.raw_matrix.length;
        const w = state.serverData.raw_matrix[0].length;
        const x = Math.min(w - 1, Math.max(0, Math.floor(rx * w)));
        const y = Math.min(h - 1, Math.max(0, Math.floor(ry * h)));
        return { x, y, value: state.serverData.raw_matrix[y][x] };
    });

    setupHover(featureCanvas, 'Feature map', (rx, ry) => {
        if (!state.serverData || !state.serverData.feature_matrix) return null;
        const h = state.serverData.feature_matrix.length;
        const w = state.serverData.feature_matrix[0].length;
        const x = Math.min(w - 1, Math.max(0, Math.floor(rx * w)));
        const y = Math.min(h - 1, Math.max(0, Math.floor(ry * h)));
        const value = state.liveFeature && state.liveFeature[y] ? state.liveFeature[y][x] : state.serverData.feature_matrix[y][x];
        return { x, y, value };
    });

    function fileToDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    upload.addEventListener('change', async e => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        state.imageDataUrl = await fileToDataUrl(file);
        equationSum.textContent = 'Image loaded. Click Run Convolution.';
        appendChatMessage('user', 'Image uploaded');
        appendChatMessage('bot', 'Nice. Now we can learn how each parameter changes the output.', true);
        if (state.guideStep <= 1) {
            askGuideStep(1);
        }
    });

    kernelSelect.addEventListener('change', () => {
        if (kernelSelect.value === 'custom') {
            customWrap.classList.remove('hidden');
        } else {
            customWrap.classList.add('hidden');
        }
        updateLiveContext();
    });

    [strideSlider, paddingSlider, filtersSlider].forEach(el => {
        el.addEventListener('input', updateSliderLabels);
    });

    runBtn.addEventListener('click', runLab);
    playBtn.addEventListener('click', playAnimation);
    pauseBtn.addEventListener('click', pauseAnimation);
    nextBtn.addEventListener('click', () => {
        pauseAnimation();
        animationStep();
    });

    if (presetEdges) {
        presetEdges.addEventListener('click', () => applyPreset('edges'));
    }
    if (presetSmooth) {
        presetSmooth.addEventListener('click', () => applyPreset('smooth'));
    }
    if (presetSharpen) {
        presetSharpen.addEventListener('click', () => applyPreset('sharpen'));
    }

    updateSliderLabels();
    updateLiveContext();

    if (chatStart) {
        chatStart.addEventListener('click', () => {
            askGuideStep(0);
        });
    }

    if (chatReset) {
        chatReset.addEventListener('click', () => {
            state.guideStep = 0;
            if (chatThread) {
                chatThread.innerHTML = '';
            }
            if (chatChoices) {
                chatChoices.innerHTML = '';
            }
            appendChatMessage('bot', 'Guide reset. We will start from image selection again.', true);
            askGuideStep(0);
        });
    }

    appendChatMessage('bot', 'Welcome. If you are new to CNNs, click Start Guided Setup and I will explain every choice and its impact.', true);
}

function initializeArchitecturesStudio() {
    const chipsContainer = document.getElementById('arch-model-chips');
    const titleEl = document.getElementById('arch-model-title');
    const taglineEl = document.getElementById('arch-model-tagline');
    const theoryEl = document.getElementById('arch-model-theory');
    const workingEl = document.getElementById('arch-model-working');
    const strengthsEl = document.getElementById('arch-model-strengths');
    const tradeoffsEl = document.getElementById('arch-model-tradeoffs');
    const useCasesEl = document.getElementById('arch-use-cases');

    const taskSelect = document.getElementById('arch-task');
    const dataSelect = document.getElementById('arch-data-size');
    const latencySelect = document.getElementById('arch-latency');
    const hardwareSelect = document.getElementById('arch-hardware');
    const generateBtn = document.getElementById('arch-generate-plan');
    const outputEl = document.getElementById('arch-plan-output');

    if (!chipsContainer || !titleEl || !theoryEl || !workingEl || !useCasesEl || !generateBtn) {
        return;
    }

    const models = [
        {
            key: 'lenet',
            name: 'LeNet-5',
            tagline: 'Classic starter CNN for low-resolution grayscale tasks.',
            theory: 'LeNet established the Conv->Pool->Conv->Pool->FC pattern. It works well when images are simple and classes are cleanly separated.',
            working: 'Small receptive fields extract local strokes, pooling reduces spatial size, dense layers classify. Low depth means easy training and interpretation.',
            strengths: ['Great for learning CNN fundamentals', 'Fast training on CPU', 'Strong baseline for MNIST-like tasks'],
            tradeoffs: ['Limited performance on complex natural images', 'Not deep enough for rich semantic abstraction'],
            useCases: [
                { title: 'Digit Recognition', text: 'Bank check digits, OCR forms, meter readings.' },
                { title: 'Embedded Prototypes', text: 'Quick proof-of-concept in low-compute environments.' }
            ]
        },
        {
            key: 'alexnet',
            name: 'AlexNet',
            tagline: 'First deep CNN breakthrough for large-scale image classification.',
            theory: 'AlexNet used deeper conv stacks, ReLU activations, and dropout to improve representation capacity and regularization.',
            working: 'Early layers capture edges/textures; deeper layers capture object parts. ReLU accelerates learning, dropout helps reduce overfitting.',
            strengths: ['Historical breakthrough', 'Clear deep-learning architecture progression', 'Good educational transition to modern CNNs'],
            tradeoffs: ['Parameter heavy by today standards', 'Less efficient than modern families'],
            useCases: [
                { title: 'General Image Classification', text: 'Educational ImageNet-like pipelines and transfer-learning demos.' },
                { title: 'Feature Extractor Baseline', text: 'Benchmarking against older deep CNN architectures.' }
            ]
        },
        {
            key: 'vgg',
            name: 'VGG16 / VGG19',
            tagline: 'Simple stacked 3x3 blocks with strong representational power.',
            theory: 'VGG showed that depth with uniform small kernels improves feature hierarchy while keeping design conceptually simple.',
            working: 'Repeated Conv(3x3)->Conv(3x3)->Pool blocks gradually expand channels and shrink spatial size, then dense classifier head.',
            strengths: ['Very clean architecture for teaching', 'Strong generic feature extractor', 'Great for visualization of learned filters'],
            tradeoffs: ['Large memory footprint', 'High compute and slower inference'],
            useCases: [
                { title: 'Transfer Learning', text: 'Feature extraction for medium-size custom datasets.' },
                { title: 'Style / Texture Tasks', text: 'Popular backbone for style-transfer and perceptual features.' }
            ]
        },
        {
            key: 'resnet',
            name: 'ResNet (18/34/50+)',
            tagline: 'Residual skip connections enable very deep networks.',
            theory: 'Residual learning lets layers model refinements over identity, reducing optimization difficulty and gradient degradation in deep nets.',
            working: 'Residual blocks compute F(x)+x where x skips across layers. This allows stable training at high depth with robust accuracy.',
            strengths: ['Excellent depth vs stability', 'Strong default for many vision tasks', 'Widely available pretrained weights'],
            tradeoffs: ['More complex than VGG/LeNet', 'Can be overkill for tiny datasets'],
            useCases: [
                { title: 'Industrial Quality Inspection', text: 'Defect detection with robust feature extraction.' },
                { title: 'Medical and Satellite Imaging', text: 'Deep feature learning on challenging textures and structures.' }
            ]
        },
        {
            key: 'efficientnet',
            name: 'EfficientNet (B0-B7)',
            tagline: 'Balanced scaling of depth, width, and resolution for efficiency.',
            theory: 'Compound scaling jointly increases depth, width, and resolution instead of scaling only one dimension.',
            working: 'Mobile-inverted blocks with squeeze-excitation are scaled systematically across model sizes for better accuracy-per-FLOP.',
            strengths: ['Excellent accuracy-efficiency trade-off', 'Great for production and mobile variants', 'Strong transfer-learning behavior'],
            tradeoffs: ['Architecture is less intuitive for beginners', 'Some variants need careful optimization settings'],
            useCases: [
                { title: 'Mobile Visual Apps', text: 'On-device classification with constrained latency budgets.' },
                { title: 'Cloud Inference Optimization', text: 'Lower serving cost per request at similar accuracy.' }
            ]
        }
    ];

    function renderModel(modelKey) {
        const model = models.find(m => m.key === modelKey) || models[0];

        titleEl.textContent = model.name;
        taglineEl.textContent = model.tagline;
        theoryEl.textContent = model.theory;
        workingEl.textContent = model.working;

        strengthsEl.innerHTML = model.strengths.map(item => `<li>${item}</li>`).join('');
        tradeoffsEl.innerHTML = model.tradeoffs.map(item => `<li>${item}</li>`).join('');

        useCasesEl.innerHTML = model.useCases.map(card => `
            <article class="arch-usecase-card">
                <h6>${card.title}</h6>
                <p>${card.text}</p>
            </article>
        `).join('');

        chipsContainer.querySelectorAll('.arch-chip').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.modelKey === model.key);
        });
    }

    chipsContainer.innerHTML = models.map(model => `
        <button type="button" class="arch-chip" data-model-key="${model.key}">${model.name}</button>
    `).join('');

    chipsContainer.querySelectorAll('.arch-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            renderModel(btn.dataset.modelKey);
        });
    });

    function recommendArchitecturePlan() {
        const task = taskSelect.value;
        const dataSize = dataSelect.value;
        const latency = latencySelect.value;
        const hardware = hardwareSelect.value;

        let base = 'ResNet18';
        let why = 'Balanced depth and training stability for many practical tasks.';
        let plan = 'Conv stem -> residual blocks -> global average pooling -> dense head';

        if (task === 'mobile' || latency === 'strict' || hardware === 'cpu') {
            base = 'EfficientNet-B0 or MobileNetV3';
            why = 'Optimized for lower latency and compute budgets.';
            plan = 'Lightweight inverted bottlenecks -> squeeze-excitation -> compact classification head';
        } else if (task === 'medical' && dataSize === 'small') {
            base = 'ResNet34 with transfer learning';
            why = 'Transfer learning helps when labeled medical data is limited.';
            plan = 'Pretrained backbone -> fine-tune top blocks -> task-specific dense head';
        } else if (task === 'classification' && dataSize === 'small') {
            base = 'VGG-style compact network';
            why = 'Simple stack is interpretable and works well for controlled datasets.';
            plan = '2-3 conv blocks -> max pooling -> dropout -> dense classifier';
        } else if (task === 'detection') {
            base = 'ResNet50 / EfficientNet backbone';
            why = 'Detection pipelines need strong multi-scale features from backbone networks.';
            plan = 'Backbone CNN -> feature pyramid -> detection head (box + class)';
        }

        const trainingTips = [
            'Start with pretrained weights whenever possible.',
            'Use data augmentation (flip, crop, color jitter) for generalization.',
            'Track train/validation loss to detect overfitting early.',
            'Tune learning rate before tuning architecture depth.'
        ];

        outputEl.innerHTML = `
            <h5>Recommended Starter Architecture</h5>
            <p><strong>Model family:</strong> ${base}</p>
            <p><strong>Why this fit:</strong> ${why}</p>
            <p><strong>Suggested block plan:</strong> ${plan}</p>
            <p><strong>Your constraints:</strong> task=${task}, data=${dataSize}, latency=${latency}, hardware=${hardware}</p>
            <p><strong>How to build your own:</strong></p>
            <ul class="arch-list">
                ${trainingTips.map(t => `<li>${t}</li>`).join('')}
            </ul>
        `;
    }

    generateBtn.addEventListener('click', recommendArchitecturePlan);
    renderModel('resnet');
}

function estimateParameters(conv1, conv2, dense, flatten) {
    const conv1Params = 3 * 3 * 1 * conv1 + conv1;
    const conv2Params = 3 * 3 * conv1 * conv2 + conv2;
    const denseParams = flatten * dense + dense;
    const outputParams = dense * 10 + 10;
    return conv1Params + conv2Params + denseParams + outputParams;
}

function createArchitectureDiagram(inputSize, conv1, conv2, dense, pool1, pool2) {
    const layers = [
        { name: 'Input', size: `${inputSize}×${inputSize}`, color: '#3b82f6' },
        { name: 'Conv1', size: `${inputSize}×${inputSize}×${conv1}`, color: '#10b981' },
        { name: 'Pool1', size: `${pool1}×${pool1}×${conv1}`, color: '#f59e0b' },
        { name: 'Conv2', size: `${pool1}×${pool1}×${conv2}`, color: '#10b981' },
        { name: 'Pool2', size: `${pool2}×${pool2}×${conv2}`, color: '#f59e0b' },
        { name: 'Dense', size: `${dense} units`, color: '#8b5cf6' }
    ];

    let html = '<div style="display: flex; align-items: center; gap: 10px;">';
    layers.forEach((layer, index) => {
        html += `
            <div style="text-align: center;">
                <div style="width: 80px; height: 80px; background-color: ${layer.color}; color: white; display: flex; align-items: center; justify-content: center; border-radius: 5px; font-weight: bold; font-size: 12px;">
                    ${layer.name}
                </div>
                <div style="font-size: 10px; color: #6b7280; margin-top: 5px;">${layer.size}</div>
            </div>
        `;
        if (index < layers.length - 1) {
            html += '<div style="font-size: 24px; color: #9ca3af;">&rarr;</div>';
        }
    });
    html += '</div>';
    return html;
}
