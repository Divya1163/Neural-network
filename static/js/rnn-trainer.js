/* RNN Trainer - Interactive visualizations and animations */

(function() {
    // Modal Management
    function initializeModals() {
        // Card click handlers - open modals
        document.querySelectorAll('.rnn-card').forEach(card => {
            card.addEventListener('click', () => {
                const targetId = card.dataset.modalTarget;
                const modal = document.getElementById(targetId);
                if (modal) {
                    modal.classList.add('active');
                }
            });
        });

        // Close button handlers
        document.querySelectorAll('.close-button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.modal').classList.remove('active');
            });
        });

        // Close on background click
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.classList.remove('active');
                }
            });
        });
    }

    // UI Elements
    const hiddenSizeSlider = document.getElementById('hidden-size');
    const learningRateSlider = document.getElementById('learning-rate');
    const epochsSlider = document.getElementById('epochs');
    
    const hiddenSizeValue = document.getElementById('hidden-size-value');
    const learningRateValue = document.getElementById('learning-rate-value');
    const epochsValue = document.getElementById('epochs-value');
    
    const trainBtn = document.getElementById('train-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const archCanvas = document.getElementById('architecture-canvas');
    const lossCanvas = document.getElementById('loss-canvas');
    const heatmapCanvas = document.getElementById('heatmap-canvas');
    const forwardPassCanvas = document.getElementById('forward-pass-canvas');
    
    const currentEpoch = document.getElementById('current-epoch');
    const currentLoss = document.getElementById('current-loss');
    const finalLoss = document.getElementById('final-loss');
    const improvement = document.getElementById('improvement');
    
    const playBtn = document.getElementById('play-forward');
    const stepBtn = document.getElementById('step-forward');
    const timestepDisplay = document.getElementById('timestep-display');
    
    let trainingData = null;
    let isTraining = false;
    let forwardPassStep = 0;
    let maxForwardSteps = 3;
    
    // Update value displays
    hiddenSizeSlider.addEventListener('input', () => {
        hiddenSizeValue.textContent = hiddenSizeSlider.value;
    });
    
    learningRateSlider.addEventListener('input', () => {
        learningRateValue.textContent = learningRateSlider.value;
    });
    
    epochsSlider.addEventListener('input', () => {
        epochsValue.textContent = epochsSlider.value;
    });
    
    // Training button
    trainBtn.addEventListener('click', startTraining);
    resetBtn.addEventListener('click', resetTraining);
    
    // Forward pass animation
    playBtn.addEventListener('click', playForwardPass);
    stepBtn.addEventListener('click', stepForwardPass);
    
    function drawArchitecture() {
        if (!archCanvas) return;
        const ctx = archCanvas.getContext('2d');
        const width = archCanvas.width;
        const height = archCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(0, 0, width, height);
        
        const hiddenSize = parseInt(hiddenSizeSlider.value);
        const centerY = height / 2;
        const layerSpacing = width / 4;
        
        // Draw layers
        const inputX = layerSpacing * 0.5;
        const hiddenX = layerSpacing * 1.5;
        const hiddenX2 = layerSpacing * 2.5;
        const outputX = layerSpacing * 3.5;
        
        // Input layer
        drawNode(ctx, inputX, centerY, 'x_t', '#60a5fa', 40);
        ctx.fillStyle = '#64748b';
        ctx.font = 'bold 12px Manrope';
        ctx.textAlign = 'center';
        ctx.fillText('Input', inputX, centerY + 40);
        
        // Hidden layer (t)
        drawNode(ctx, hiddenX, centerY, 'h_t', '#34d399', 50);
        ctx.fillText('Hidden (t)', hiddenX, centerY + 45);
        
        // Hidden layer (t-1) for recurrence
        ctx.globalAlpha = 0.6;
        drawNode(ctx, hiddenX2, centerY - 80, 'h_{t-1}', '#34d399', 45);
        ctx.globalAlpha = 1;
        ctx.fillText('Hidden (t-1)', hiddenX2, centerY - 80 + 40);
        
        // Output layer
        drawNode(ctx, outputX, centerY, 'y_t', '#a78bfa', 40);
        ctx.fillStyle = '#64748b';
        ctx.fillText('Output', outputX, centerY + 40);
        
        // Draw connections
        // Input -> Hidden
        drawArrow(ctx, inputX + 40, centerY, hiddenX - 50, centerY, '#667eea', 'W_xh');
        
        // Hidden (t-1) -> Hidden (t)
        drawArrow(ctx, hiddenX2, centerY - 80 + 45, hiddenX - 20, centerY - 45, '#f97316', 'W_hh');
        
        // Hidden -> Output
        drawArrow(ctx, hiddenX + 50, centerY, outputX - 40, centerY, '#667eea', 'W_hy');
        
        // Recurrent loop (self-connection)
        drawRecurrentLoop(ctx, hiddenX, centerY);
    }
    
    function drawNode(ctx, x, y, label, color, radius) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px Manrope';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x, y);
    }
    
    function drawArrow(ctx, fromX, fromY, toX, toY, color, label) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        
        // Arrowhead
        const angle = Math.atan2(toY - fromY, toX - fromX);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - 10 * Math.cos(angle - Math.PI / 6), toY - 10 * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - 10 * Math.cos(angle + Math.PI / 6), toY - 10 * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
        
        // Label
        const midX = (fromX + toX) / 2;
        const midY = (fromY + toY) / 2;
        ctx.fillStyle = color;
        ctx.font = 'bold 11px Manrope';
        ctx.textAlign = 'center';
        ctx.fillText(label, midX, midY - 10);
    }
    
    function drawRecurrentLoop(ctx, x, y) {
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([5, 5]);
        
        const radius = 35;
        ctx.beginPath();
        ctx.arc(x, y, radius, Math.PI / 4, -Math.PI / 4, false);
        ctx.stroke();
        
        // Arrowhead
        ctx.setLineDash([]);
        ctx.fillStyle = '#f97316';
        ctx.beginPath();
        ctx.moveTo(x + radius * Math.cos(-Math.PI / 4), y + radius * Math.sin(-Math.PI / 4));
        ctx.lineTo(x + radius * Math.cos(-Math.PI / 4) - 8, y + radius * Math.sin(-Math.PI / 4) - 8);
        ctx.lineTo(x + radius * Math.cos(-Math.PI / 4) + 8, y + radius * Math.sin(-Math.PI / 4) - 8);
        ctx.closePath();
        ctx.fill();
    }
    
    function drawLossCurve(losses) {
        const ctx = lossCanvas.getContext('2d');
        const width = lossCanvas.width;
        const height = lossCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (losses.length === 0) return;
        
        const maxLoss = Math.max(...losses);
        const minLoss = Math.min(...losses);
        const range = maxLoss - minLoss || 1;
        
        const padding = 30;
        const graphWidth = width - padding * 2;
        const graphHeight = height - padding * 2;
        
        // Background
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(padding, padding, graphWidth, graphHeight);
        
        // Grid lines
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = padding + (graphHeight / 5) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(padding + graphWidth, y);
            ctx.stroke();
        }
        
        // Plot line
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        
        for (let i = 0; i < losses.length; i++) {
            const x = padding + (graphWidth / (losses.length - 1 || 1)) * i;
            const y = padding + graphHeight - ((losses[i] - minLoss) / range) * graphHeight;
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        
        // Axes
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, padding + graphHeight);
        ctx.lineTo(padding + graphWidth, padding + graphHeight);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = '#374151';
        ctx.font = '11px Manrope';
        ctx.textAlign = 'right';
        ctx.fillText('Loss', padding - 10, padding - 5);
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', padding + graphWidth / 2, height - 5);
    }
    
    function drawHeatmap(losses) {
        const ctx = heatmapCanvas.getContext('2d');
        const width = heatmapCanvas.width;
        const height = heatmapCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (losses.length === 0) return;
        
        const maxLoss = Math.max(...losses);
        const minLoss = Math.min(...losses);
        const range = maxLoss - minLoss || 1;
        
        const cellWidth = width / Math.min(losses.length, 50);
        const cellHeight = height / 8;
        
        // Draw heatmap
        for (let i = 0; i < Math.min(losses.length, 50); i++) {
            const normalized = (losses[i * Math.floor(losses.length / 50) || i] - minLoss) / range;
            const hue = (1 - normalized) * 240;
            
            ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
            ctx.fillRect(i * cellWidth, 0, cellWidth, height);
        }
        
        // Border
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, width, height);
    }
    
    function drawForwardPassAnimation() {
        const ctx = forwardPassCanvas.getContext('2d');
        const width = forwardPassCanvas.width;
        const height = forwardPassCanvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(0, 0, width, height);
        
        const timesteps = 3;
        const spacing = width / (timesteps + 1);
        const centerY = height / 2;
        const layerSpacing = 100;
        
        for (let t = 0; t < timesteps; t++) {
            const x = spacing * (t + 1);
            const isActive = t <= forwardPassStep;
            const isCurrentStep = t === forwardPassStep;
            
            // Input
            const inputY = centerY - layerSpacing;
            ctx.fillStyle = isActive ? '#60a5fa' : '#cbd5e1';
            ctx.beginPath();
            ctx.arc(x, inputY, 20, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = isCurrentStep ? '#fff' : '#9ca3af';
            ctx.font = 'bold 12px Manrope';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`x${t+1}`, x, inputY);
            
            // Hidden
            const hiddenY = centerY;
            ctx.fillStyle = isActive ? '#34d399' : '#cbd5e1';
            ctx.beginPath();
            ctx.arc(x, hiddenY, 25, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = isCurrentStep ? '#fff' : '#9ca3af';
            ctx.fillText(`h${t+1}`, x, hiddenY);
            
            // Output
            const outputY = centerY + layerSpacing;
            ctx.fillStyle = isActive ? '#a78bfa' : '#cbd5e1';
            ctx.beginPath();
            ctx.arc(x, outputY, 20, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = isCurrentStep ? '#fff' : '#9ca3af';
            ctx.fillText(`y${t+1}`, x, outputY);
            
            // Connections (if not first timestep)
            if (t > 0 && isActive) {
                ctx.strokeStyle = isCurrentStep ? '#f97316' : '#667eea';
                ctx.lineWidth = 2;
                
                // From h_{t-1} to h_t
                ctx.beginPath();
                ctx.moveTo(x - spacing, centerY);
                ctx.lineTo(x, centerY);
                ctx.stroke();
                
                if (isCurrentStep) {
                    ctx.fillStyle = '#f97316';
                    ctx.font = '10px Manrope';
                    ctx.fillText('W_hh', x - spacing / 2, centerY - 15);
                }
            }
            
            // Input -> Hidden
            if (isActive) {
                ctx.strokeStyle = isCurrentStep ? '#f97316' : '#667eea';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, inputY + 20);
                ctx.lineTo(x, hiddenY - 25);
                ctx.stroke();
            }
            
            // Hidden -> Output
            if (isActive) {
                ctx.strokeStyle = isCurrentStep ? '#f97316' : '#667eea';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, hiddenY + 25);
                ctx.lineTo(x, outputY - 20);
                ctx.stroke();
            }
        }
        
        timestepDisplay.textContent = `Timestep: ${forwardPassStep + 1}/${timesteps}`;
    }
    
    function playForwardPass() {
        if (forwardPassStep < maxForwardSteps) {
            forwardPassStep++;
            drawForwardPassAnimation();
            
            if (forwardPassStep >= maxForwardSteps) {
                playBtn.disabled = true;
            }
        }
    }
    
    function stepForwardPass() {
        playForwardPass();
    }
    
    function resetForwardPass() {
        forwardPassStep = 0;
        playBtn.disabled = false;
        drawForwardPassAnimation();
    }
    
    async function startTraining() {
        if (isTraining) return;
        
        isTraining = true;
        trainBtn.disabled = true;
        resetBtn.disabled = true;
        
        const hiddenSize = parseInt(hiddenSizeSlider.value);
        const learningRate = parseFloat(learningRateSlider.value);
        const epochs = parseInt(epochsSlider.value);
        
        try {
            const response = await fetch('/api/rnn/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    hidden_size: hiddenSize,
                    learning_rate: learningRate,
                    num_epochs: epochs
                })
            });
            
            if (!response.ok) throw new Error('Training failed');
            
            const data = await response.json();
            trainingData = data;
            
            // Animate loss curve
            animateLossCurve(data.losses);
            
            // Update stats
            finalLoss.textContent = data.final_loss.toFixed(6);
            improvement.textContent = data.improvement.toFixed(1) + '%';
            
            resetForwardPass();
            
        } catch (error) {
            console.error('Training error:', error);
            alert('Training failed: ' + error.message);
        } finally {
            isTraining = false;
            trainBtn.disabled = false;
            resetBtn.disabled = false;
        }
    }
    
    function animateLossCurve(losses) {
        let currentIndex = 0;
        const animationInterval = setInterval(() => {
            if (currentIndex <= losses.length) {
                const currentLosses = losses.slice(0, currentIndex);
                drawLossCurve(currentLosses);
                drawHeatmap(currentLosses);
                
                if (currentIndex > 0) {
                    currentEpoch.textContent = currentIndex;
                    currentLoss.textContent = currentLosses[currentIndex - 1].toFixed(6);
                }
                
                currentIndex++;
            } else {
                clearInterval(animationInterval);
            }
        }, 30);
    }
    
    function resetTraining() {
        trainingData = null;
        forwardPassStep = 0;
        playBtn.disabled = false;
        currentEpoch.textContent = '0';
        currentLoss.textContent = '--';
        finalLoss.textContent = '--';
        improvement.textContent = '--';
        
        const ctx1 = lossCanvas.getContext('2d');
        ctx1.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
        
        const ctx2 = heatmapCanvas.getContext('2d');
        ctx2.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        
        drawForwardPassAnimation();
    }
    
    // Initialize
    function init() {
        initializeModals();
        drawArchitecture();
        resetForwardPass();
    }
    
    init();
})();
