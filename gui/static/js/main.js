// Main JavaScript for Molecular Design System GUI

class MolecularDesignUI {
    constructor() {
        this.eventSource = null;
        this.isRunning = false;
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        // Input elements
        this.densityInput = document.getElementById('density');
        this.velocityInput = document.getElementById('velocity');
        this.pressureInput = document.getElementById('pressure');
        this.hfInput = document.getElementById('hf');
        this.enableRagInput = document.getElementById('enable-rag');
        this.beamWidthInput = document.getElementById('beam-width');
        this.topKInput = document.getElementById('top-k');
        this.maxIterInput = document.getElementById('max-iter');
        this.mapeThresholdInput = document.getElementById('mape-threshold');
        this.runButton = document.getElementById('run-btn');
        this.statusMessage = document.getElementById('status-message');

        // Display elements
        this.iterationCount = document.getElementById('iteration-count');
        this.candidateCount = document.getElementById('candidate-count');
        this.bestScore = document.getElementById('best-score');
        this.bestMAPE = document.getElementById('best-mape');
        this.bestFeasibility = document.getElementById('best-feasibility');
        this.candidatesContainer = document.getElementById('candidates-container');
        this.seedMolecule = document.getElementById('seed-molecule');
        this.targetDisplay = document.getElementById('target-display');
        this.bestMolecule = document.getElementById('best-molecule');

        // Create popup element
        this.popup = this.createPopupElement();
        this.targetProperties = null;
    }

    createPopupElement() {
        const popup = document.createElement('div');
        popup.className = 'candidate-popup';
        popup.id = 'candidate-popup';
        document.body.appendChild(popup);

        // Close popup when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.candidate-card') && !e.target.closest('.candidate-popup')) {
                popup.classList.remove('show');
            }
        });

        return popup;
    }

    attachEventListeners() {
        this.runButton.addEventListener('click', () => this.startOptimization());
    }

    async startOptimization() {
        if (this.isRunning) return;

        // Collect parameters
        const params = {
            density: parseFloat(this.densityInput.value),
            velocity: parseFloat(this.velocityInput.value),
            pressure: parseFloat(this.pressureInput.value),
            hf: parseFloat(this.hfInput.value),
            enable_rag: this.enableRagInput.checked,
            beam_width: parseInt(this.beamWidthInput.value),
            top_k: parseInt(this.topKInput.value),
            max_iter: parseInt(this.maxIterInput.value),
            mape_threshold: parseFloat(this.mapeThresholdInput.value)
        };

        // Validate inputs
        if (!this.validateInputs(params)) {
            this.showStatus('Please enter valid parameters', 'error');
            return;
        }

        // Update UI
        this.isRunning = true;
        this.runButton.disabled = true;
        this.runButton.querySelector('.button-text').style.display = 'none';
        this.runButton.querySelector('.button-loader').style.display = 'inline-block';
        this.showStatus('Starting optimization...', 'info');

        // Reset displays
        this.resetDisplays();

        try {
            // Start optimization
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error('Failed to start optimization');
            }

            // Start listening for updates
            this.connectEventSource();

        } catch (error) {
            console.error('Error starting optimization:', error);
            this.showStatus('Error: ' + error.message, 'error');
            this.resetUI();
        }
    }

    connectEventSource() {
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource('/api/progress');

        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleUpdate(data);
        };

        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            this.eventSource.close();
            this.resetUI();
        };
    }

    handleUpdate(data) {
        console.log('Update received:', data.type);

        switch (data.type) {
            case 'status':
                this.showStatus(data.message, 'info');
                break;

            case 'target':
                this.displayTarget(data.properties);
                break;

            case 'seed':
                this.displaySeedMolecule(data);
                break;

            case 'iteration':
                this.displayIteration(data);
                break;

            case 'best':
                this.displayBestMolecule(data);
                break;

            case 'complete':
                this.showStatus('Optimization complete!', 'info');
                this.resetUI();
                break;

            case 'error':
                this.showStatus('Error: ' + data.message, 'error');
                this.resetUI();
                break;

            case 'heartbeat':
                // Keep-alive, no action needed
                break;
        }
    }

    displayTarget(properties) {
        this.targetProperties = properties; // Store for MAPE calculation

        this.targetDisplay.innerHTML = `
            <div class="property-row">
                <span class="property-label">Density</span>
                <span class="property-value">${properties['Density'].toFixed(2)} g/cm³</span>
            </div>
            <div class="property-row">
                <span class="property-label">Det. Velocity</span>
                <span class="property-value">${properties['Det Velocity'].toFixed(0)} m/s</span>
            </div>
            <div class="property-row">
                <span class="property-label">Det. Pressure</span>
                <span class="property-value">${properties['Det Pressure'].toFixed(1)} GPa</span>
            </div>
            <div class="property-row">
                <span class="property-label">Hf solid</span>
                <span class="property-value">${properties['Hf solid'].toFixed(1)} kJ/mol</span>
            </div>
        `;
    }

    displaySeedMolecule(data) {
        this.seedMolecule.innerHTML = `
            <img src="${data.image}" alt="Seed Molecule" class="molecule-img">
            <div class="properties-grid">
                <div class="property-row">
                    <span class="property-label">Score</span>
                    <span class="property-value">${data.score.toFixed(4)}</span>
                </div>
                <div class="property-row">
                    <span class="property-label">Feasibility</span>
                    <span class="property-value">${((1 - data.feasibility) * 100).toFixed(0)}%</span>
                </div>
                ${this.formatProperties(data.properties)}
            </div>
        `;
    }

    displayIteration(data) {
        this.iterationCount.textContent = data.iteration;
        this.candidateCount.textContent = data.candidates.length;

        const topK = data.top_k || parseInt(this.topKInput.value);

        // Sort candidates by MAPE (lower is better) before displaying
        const sortedCandidates = [...data.candidates].sort((a, b) => a.mape - b.mape);

        // Create iteration row with horizontal scroll
        const rowHTML = `
            <div class="iteration-row fade-in">
                <div class="iteration-header">
                    <h3 class="iteration-title">Iteration ${data.iteration}</h3>
                    <span class="iteration-stats">${sortedCandidates.length} candidates | Top ${topK} by MAPE</span>
                </div>
                <div class="candidates-scroll">
                    ${sortedCandidates.map((cand, i) => {
            const isSelected = i < topK;
            return `
                            <div class="candidate-card ${isSelected ? 'selected' : ''}" 
                                 data-index="${i}"
                                 data-score="${cand.score}"
                                 data-mape="${cand.mape}"
                                 data-feasibility="${cand.feasibility}"
                                 data-props='${JSON.stringify(cand.properties)}'>
                                ${cand.image ? `<img src="${cand.image}" class="candidate-img" alt="Candidate ${i + 1}">` : ''}
                                <div class="candidate-score">MAPE: ${cand.mape.toFixed(2)}%</div>
                                <div class="candidate-feasibility">${((1 - cand.feasibility) * 100).toFixed(0)}%</div>
                            </div>
                        `;
        }).join('')}
                </div>
            </div>
        `;

        // Prepend new iteration (most recent on top)
        this.candidatesContainer.innerHTML = rowHTML + this.candidatesContainer.innerHTML;

        // Attach click handlers for popups
        this.attachCandidateClickHandlers();
    }

    attachCandidateClickHandlers() {
        const cards = document.querySelectorAll('.candidate-card');
        cards.forEach(card => {
            card.addEventListener('click', (e) => {
                e.stopPropagation();
                this.showCandidatePopup(card, e);
            });
        });
    }

    showCandidatePopup(card, event) {
        const score = parseFloat(card.dataset.score);
        const feasibility = parseFloat(card.dataset.feasibility);
        const properties = JSON.parse(card.dataset.props);
        const isSelected = card.classList.contains('selected');

        // Calculate MAPE
        const mape = this.calculateMAPE(properties);

        // Position popup next to card
        const rect = card.getBoundingClientRect();
        const popupX = rect.right + 10;
        const popupY = rect.top;

        this.popup.style.left = `${popupX}px`;
        this.popup.style.top = `${popupY}px`;

        // Populate popup content
        this.popup.innerHTML = `
            <div class="popup-header">
                Candidate Details ${isSelected ? '✓' : ''}
            </div>
            <div class="popup-section">
                <div class="popup-label">Score</div>
                <div class="popup-value">${score.toFixed(4)}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Feasibility</div>
                <div class="popup-value">${((1 - feasibility) * 100).toFixed(1)}%</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">MAPE (vs Target)</div>
                <div class="popup-value">${mape.toFixed(2)}%</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Properties</div>
                <div class="popup-properties">
                    <div class="popup-prop-row">
                        <span class="popup-prop-label">Density:</span>
                        <span class="popup-prop-value">${properties['Density'].toFixed(3)} g/cm³</span>
                    </div>
                    <div class="popup-prop-row">
                        <span class="popup-prop-label">Det. Velocity:</span>
                        <span class="popup-prop-value">${properties['Det Velocity'].toFixed(1)} m/s</span>
                    </div>
                    <div class="popup-prop-row">
                        <span class="popup-prop-label">Det. Pressure:</span>
                        <span class="popup-prop-value">${properties['Det Pressure'].toFixed(2)} GPa</span>
                    </div>
                    <div class="popup-prop-row">
                        <span class="popup-prop-label">Hf solid:</span>
                        <span class="popup-prop-value">${properties['Hf solid'].toFixed(2)} kJ/mol</span>
                    </div>
                </div>
            </div>
        `;

        this.popup.classList.add('show');

        // Adjust if popup goes off screen
        setTimeout(() => {
            const popupRect = this.popup.getBoundingClientRect();
            if (popupRect.right > window.innerWidth) {
                this.popup.style.left = `${rect.left - popupRect.width - 10}px`;
            }
            if (popupRect.bottom > window.innerHeight) {
                this.popup.style.top = `${window.innerHeight - popupRect.height - 10}px`;
            }
        }, 10);
    }

    calculateMAPE(properties) {
        if (!this.targetProperties) return 0;

        const propertyNames = ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid'];
        let totalError = 0;
        let count = 0;

        propertyNames.forEach(prop => {
            if (this.targetProperties[prop] !== undefined && properties[prop] !== undefined) {
                const target = this.targetProperties[prop];
                const pred = properties[prop];
                // MAPE: |actual - predicted| / |actual| * 100
                if (Math.abs(target) > 1e-10) {
                    const percentError = Math.abs(target - pred) / Math.abs(target) * 100;
                    totalError += percentError;
                    count++;
                }
            }
        });

        return count > 0 ? totalError / count : 0;
    }

    displayBestMolecule(data) {
        this.bestScore.textContent = data.score.toFixed(4);

        // Calculate and display MAPE
        const mape = this.calculateMAPE(data.properties);
        this.bestMAPE.textContent = mape.toFixed(2) + '%';

        // Display feasibility (inverted: 100% = most feasible, 0% = least feasible)
        this.bestFeasibility.textContent = ((1 - data.feasibility) * 100).toFixed(0) + '%';

        this.bestMolecule.innerHTML = `
            <img src="${data.image}" alt="Best Molecule" class="molecule-img">
            <div class="score-badge">Score: ${data.score.toFixed(4)}</div>
            <div class="feasibility-badge">Feasibility: ${((1 - data.feasibility) * 100).toFixed(0)}%</div>
            <div class="properties-grid">
                ${this.formatProperties(data.properties)}
            </div>
            <p style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted); word-break: break-all;">
                SMILES: ${data.smiles}
            </p>
        `;
    }

    formatProperties(properties) {
        return `
            <div class="property-row">
                <span class="property-label">Density</span>
                <span class="property-value">${properties['Density'].toFixed(3)} g/cm³</span>
            </div>
            <div class="property-row">
                <span class="property-label">Det. Velocity</span>
                <span class="property-value">${properties['Det Velocity'].toFixed(1)} m/s</span>
            </div>
            <div class="property-row">
                <span class="property-label">Det. Pressure</span>
                <span class="property-value">${properties['Det Pressure'].toFixed(2)} GPa</span>
            </div>
            <div class="property-row">
                <span class="property-label">Hf solid</span>
                <span class="property-value">${properties['Hf solid'].toFixed(2)} kJ/mol</span>
            </div>
        `;
    }

    resetDisplays() {
        this.iterationCount.textContent = '0';
        this.candidateCount.textContent = '0';
        this.bestScore.textContent = '-';
        this.candidatesContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">⏳</div>
                <p>Running optimization...</p>
            </div>
        `;
    }

    resetUI() {
        this.isRunning = false;
        this.runButton.disabled = false;
        this.runButton.querySelector('.button-text').style.display = 'inline';
        this.runButton.querySelector('.button-loader').style.display = 'none';

        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    validateInputs(params) {
        return params.density > 0 &&
            params.velocity > 0 &&
            params.pressure > 0 &&
            params.beam_width > 0 &&
            params.max_iter > 0;
    }

    showStatus(message, type) {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MolecularDesignUI();
});
