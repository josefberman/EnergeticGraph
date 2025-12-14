// ===== STATE =====
let socket;
let isRunning = false;
let statusCheckInterval = null;
let currentErrorMetric = 'mape'; // Track current error metric

// ===== DOM ELEMENTS =====
const elements = {
    // Upload
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadedFileName: document.getElementById('uploadedFileName'),

    // Settings
    useRAG: document.getElementById('useRAG'),
    errorMetric: document.getElementById('errorMetric'),
    beamWidth: document.getElementById('beamWidth'),
    maxIterations: document.getElementById('maxIterations'),
    proceedK: document.getElementById('proceedK'),
    startMode: document.getElementById('startMode'),
    smilesInputGroup: document.getElementById('smilesInputGroup'),
    customSmiles: document.getElementById('customSmiles'),

    // Buttons
    runBtn: document.getElementById('runBtn'),
    resetBtn: document.getElementById('resetBtn'),

    // States
    emptyState: document.getElementById('emptyState'),
    loadingState: document.getElementById('loadingState'),
    resultsDisplay: document.getElementById('resultsDisplay'),
    statusText: document.getElementById('statusText'),
    progressFill: document.getElementById('progressFill'),

    // Results
    scoreDisplay: document.getElementById('scoreDisplay'),
    startMolImg: document.getElementById('startMolImg'),
    bestMolImg: document.getElementById('bestMolImg'),
    startSmiles: document.getElementById('startSmiles'),
    bestSmiles: document.getElementById('bestSmiles'),
    targetProps: document.getElementById('targetProps'),
    bestProps: document.getElementById('bestProps'),
    iterationHistory: document.getElementById('iterationHistory'),
    ragSection: document.getElementById('ragSection'),
    ragContent: document.getElementById('ragContent'),

    // Toast
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage')
};

// ===== INITIALIZE =====
function init() {
    setupSocket();
    setupEventListeners();
}

// ===== SOCKET.IO =====
function setupSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('WebSocket connected');
    });

    socket.on('status_update', (data) => {
        if (data.status) {
            elements.statusText.textContent = data.status;
        }
    });

    socket.on('optimization_complete', (data) => {
        if (data.success) {
            showToast('Optimization complete!', 'success');
            loadResults();
        } else {
            showToast(`Error: ${data.error}`, 'error');
            showEmptyState();
        }
        isRunning = false;
        updateRunButton();
        stopStatusCheck();
    });
}

// ===== EVENT LISTENERS =====
function setupEventListeners() {
    // File upload
    elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadZone.addEventListener('dragover', handleDragOver);
    elements.uploadZone.addEventListener('dragleave', handleDragLeave);
    elements.uploadZone.addEventListener('drop', handleDrop);

    // Start mode change
    elements.startMode.addEventListener('change', () => {
        if (elements.startMode.value === 'custom') {
            elements.smilesInputGroup.classList.remove('hidden');
        } else {
            elements.smilesInputGroup.classList.add('hidden');
        }
    });

    // Buttons
    elements.runBtn.addEventListener('click', handleRun);
    elements.resetBtn.addEventListener('click', handleReset);
}

// ===== FILE UPLOAD =====
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (file) {
        uploadFile(file);
    }
}

async function uploadFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            elements.uploadedFileName.textContent = `✓ ${data.filename}`;
            elements.uploadedFileName.style.color = 'var(--success)';
            showToast('File uploaded successfully', 'success');
        } else {
            showToast(`Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Upload error: ${error.message}`, 'error');
    }
}

// ===== OPTIMIZATION =====
async function handleRun() {
    if (isRunning) return;

    // Store current error metric
    currentErrorMetric = elements.errorMetric.value;

    const config = {
        use_rag: elements.useRAG.checked,
        error_metric: currentErrorMetric,
        beam_width: parseInt(elements.beamWidth.value),
        max_iterations: parseInt(elements.maxIterations.value),
        proceed_k: parseInt(elements.proceedK.value),
        starting_smiles: elements.startMode.value === 'custom' ? elements.customSmiles.value : ''
    };

    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            isRunning = true;
            updateRunButton();
            showLoadingState();
            startStatusCheck();
            showToast('Optimization started', 'success');
        } else {
            showToast(`Failed to start: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

function startStatusCheck() {
    statusCheckInterval = setInterval(checkStatus, 2000);
}

function stopStatusCheck() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (!data.running && data.has_results) {
            stopStatusCheck();
            loadResults();
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

async function loadResults() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showToast(`Error loading results: ${data.error}`, 'error');
            showEmptyState();
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        showEmptyState();
    }
}

function displayResults(data) {
    // Update scores
    let scoreHtml = '';
    if (data.best_score !== null && data.best_score !== undefined) {
        // Format score based on metric type
        const formattedScore = currentErrorMetric === 'mape'
            ? `${(data.best_score * 100).toFixed(2)}%`
            : data.best_score.toFixed(6);
        const metricLabel = currentErrorMetric === 'mape' ? 'MAPE' : 'MSE';
        scoreHtml = `<div class="score-main">Best ${metricLabel}: ${formattedScore}</div>`;

        const components = [];
        if (data.best_prop_error !== null) {
            const formattedPropError = currentErrorMetric === 'mape'
                ? `${(data.best_prop_error * 100).toFixed(2)}%`
                : data.best_prop_error.toFixed(6);
            components.push(`Prop Error: ${formattedPropError}`);
        }
        if (data.best_feasibility_score !== null) {
            components.push(`Feasibility: ${data.best_feasibility_score.toFixed(3)}`);
        }
        if (components.length > 0) {
            scoreHtml += `<div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem;">${components.join(' | ')}</div>`;
        }
    }
    elements.scoreDisplay.innerHTML = scoreHtml;

    // Update molecules
    elements.startMolImg.src = data.start_molecule.image;
    elements.bestMolImg.src = data.best_molecule.image;
    elements.startSmiles.textContent = data.start_molecule.smiles;
    elements.bestSmiles.textContent = data.best_molecule.smiles;

    // Update properties
    elements.targetProps.innerHTML = createPropsTable(data.target_properties);
    elements.bestProps.innerHTML = createPropsTable(data.best_properties);

    // Update iteration history
    elements.iterationHistory.innerHTML = createIterationHistory(data.search_history);

    // Update RAG trace
    if (data.rag_trace) {
        elements.ragSection.classList.remove('hidden');
        elements.ragContent.innerHTML = createRagTrace(data.rag_trace);
    } else {
        elements.ragSection.classList.add('hidden');
    }

    showResultsDisplay();
}

function createPropsTable(props) {
    if (!props || Object.keys(props).length === 0) {
        return '<p style="color: var(--text-secondary); padding: 1rem;">No properties available</p>';
    }

    const rows = Object.entries(props).map(([key, value]) =>
        `<tr><td><strong>${key}</strong></td><td>${typeof value === 'number' ? value.toFixed(4) : value}</td></tr>`
    ).join('');

    return `<table><tbody>${rows}</tbody></table>`;
}

function createIterationHistory(history) {
    if (!history || history.length === 0) {
        return '<p style="color: var(--text-secondary);">No iteration history available</p>';
    }

    return history.map(iteration => {
        const candidates = iteration.candidates.map(c => {
            // Format error based on metric type
            const formattedError = c.prop_error !== null
                ? (currentErrorMetric === 'mape'
                    ? `${(c.prop_error * 100).toFixed(2)}%`
                    : c.prop_error.toFixed(6))
                : null;

            return `
            <div class="candidate-card">
                <img src="${c.image}" alt="Candidate molecule">
                <div class="candidate-smiles">${c.smiles.substring(0, 60)}${c.smiles.length > 60 ? '...' : ''}</div>
                ${c.feasibility_score !== null ? `<div class="candidate-metric">Feasibility: <strong>${c.feasibility_score.toFixed(3)}</strong></div>` : ''}
                ${formattedError !== null ? `<div class="candidate-metric">Error (${currentErrorMetric.toUpperCase()}): <strong>${formattedError}</strong></div>` : ''}
            </div>
        `;
        }).join('');

        return `
            <div class="iteration-block">
                <div class="iteration-header" onclick="toggleIteration(this)">
                    <span>Iteration ${iteration.iteration} — ${iteration.candidates.length} candidates</span>
                    <span>▼</span>
                </div>
                <div class="iteration-content">
                    ${candidates}
                </div>
            </div>
        `;
    }).join('');
}

function toggleIteration(header) {
    const content = header.nextElementSibling;
    const arrow = header.querySelector('span:last-child');

    if (content.classList.contains('expanded')) {
        content.classList.remove('expanded');
        arrow.textContent = '▼';
    } else {
        content.classList.add('expanded');
        arrow.textContent = '▲';
    }
}

function createRagTrace(trace) {
    const titles = (trace.retrieved_titles || []).map(t => `<li>${t}</li>`).join('');

    return `
        <div style="color: var(--text-secondary);">
            <p><strong>Query:</strong> ${trace.query || 'N/A'}</p>
            <p><strong>Retrieved articles:</strong> ${trace.retrieved_count || 0}</p>
            ${titles ? `<ul style="margin-top: 1rem;">${titles}</ul>` : ''}
            <p style="margin-top: 1rem;">
                Names extracted: <strong>${trace.names_extracted || 0}</strong> |
                Names converted: <strong>${trace.names_converted || 0}</strong> |
                SMILES extracted: <strong>${trace.smiles_extracted || 0}</strong> |
                Candidates scored: <strong>${trace.candidates_scored || 0}</strong>
            </p>
            ${trace.fallback_used ? '<p style="color: var(--warning); margin-top: 1rem;">⚠ RAG yielded no molecules; fell back to local CSV</p>' : ''}
        </div>
    `;
}

// ===== RESET =====
async function handleReset() {
    if (confirm('Are you sure you want to reset? This will clear all data and stop any running optimization.')) {
        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            const data = await response.json();

            if (data.success) {
                // Reset UI
                elements.uploadedFileName.textContent = 'No file uploaded';
                elements.uploadedFileName.style.color = 'var(--text-secondary)';
                elements.useRAG.checked = false;
                elements.errorMetric.value = 'mape';
                elements.beamWidth.value = 5;
                elements.maxIterations.value = 8;
                elements.proceedK.value = 3;
                elements.startMode.value = 'samples';
                elements.smilesInputGroup.classList.add('hidden');
                elements.customSmiles.value = '';

                isRunning = false;
                updateRunButton();
                showEmptyState();
                showToast('Reset complete', 'success');
            }
        } catch (error) {
            showToast(`Reset error: ${error.message}`, 'error');
        }
    }
}

// ===== UI STATE =====
function showEmptyState() {
    elements.emptyState.classList.remove('hidden');
    elements.loadingState.classList.add('hidden');
    elements.resultsDisplay.classList.add('hidden');
}

function showLoadingState() {
    elements.emptyState.classList.add('hidden');
    elements.loadingState.classList.remove('hidden');
    elements.resultsDisplay.classList.add('hidden');
}

function showResultsDisplay() {
    elements.emptyState.classList.add('hidden');
    elements.loadingState.classList.add('hidden');
    elements.resultsDisplay.classList.remove('hidden');
}

function updateRunButton() {
    if (isRunning) {
        elements.runBtn.disabled = true;
        elements.runBtn.querySelector('.btn-text').textContent = 'Running...';
        elements.runBtn.querySelector('.btn-icon').textContent = '⏸';
    } else {
        elements.runBtn.disabled = false;
        elements.runBtn.querySelector('.btn-text').textContent = 'Run Optimization';
        elements.runBtn.querySelector('.btn-icon').textContent = '▶';
    }
}

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = 'success') {
    elements.toastMessage.textContent = message;
    elements.toast.className = `toast ${type}`;
    elements.toast.classList.remove('hidden');

    setTimeout(() => {
        elements.toast.classList.add('hidden');
    }, 4000);
}

// ===== START APP =====
document.addEventListener('DOMContentLoaded', init);
