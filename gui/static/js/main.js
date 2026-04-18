/* Energetic Designer — vanilla JS front-end for the Flask SSE backend. */

(() => {
    const $ = (id) => document.getElementById(id);

    const state = {
        es: null,
        running: false,
        iteration: 0,
        maxIter: 10,
        seedProperties: null,
        targetProperties: null,
    };

    const els = {
        density: $('density'),
        velocity: $('velocity'),
        pressure: $('pressure'),
        hf: $('hf'),
        beamWidth: $('beam-width'),
        topK: $('top-k'),
        maxIter: $('max-iter'),
        mape: $('mape-threshold'),
        enableRag: $('enable-rag'),
        runBtn: $('run-btn'),
        stopBtn: $('stop-btn'),
        statusMsg: $('status-message'),
        iterCount: $('iteration-count'),
        candCount: $('candidate-count'),
        bestMape: $('best-mape'),
        bestFeas: $('best-feasibility'),
        bestScore: $('best-score'),
        progressFill: $('progress-fill'),
        progressBar: document.querySelector('.progress-bar'),
        seedMol: $('seed-molecule'),
        bestMol: $('best-molecule'),
        target: $('target-display'),
        candidates: $('candidates-container'),
        candSub: $('candidates-sub'),
        compSub: $('comparison-sub'),
        progressSub: $('progress-subtitle'),
        ragPill: $('rag-pill'),
        runPill: $('run-pill'),
        drawer: $('candidate-drawer'),
        drawerBody: $('drawer-body'),
        drawerClose: $('drawer-close'),
    };

    // ---------------------------------------------------------------- helpers
    const fmt = (v, digits = 2) => {
        if (v === null || v === undefined || Number.isNaN(v)) return '–';
        if (typeof v !== 'number') return String(v);
        return v.toFixed(digits);
    };

    const fmtProp = (key, value) => {
        if (value === null || value === undefined) return '–';
        const units = {
            'Density': 'g/cm³',
            'Det Velocity': 'm/s',
            'Det Pressure': 'GPa',
            'Hf solid': 'kJ/mol',
        };
        const digits = key === 'Det Velocity' ? 0 : 2;
        return `${value.toFixed(digits)} ${units[key] || ''}`.trim();
    };

    const mapeBadgeClass = (mape) => {
        if (mape === null || mape === undefined) return '';
        if (mape < 5) return 'mape-good';
        if (mape < 15) return 'mape-ok';
        return 'mape-bad';
    };

    const PROP_KEYS = ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid'];

    const molPropertyTable = (props, { target = null, baseline = null } = {}) => {
        const rows = PROP_KEYS.map((key) => {
            const cell = fmtProp(key, props?.[key]);
            let delta = '';
            if (target && baseline && props?.[key] !== undefined) {
                const baseErr = Math.abs(baseline[key] - target[key]);
                const currErr = Math.abs(props[key] - target[key]);
                if (baseErr > 0) {
                    const improvement = (baseErr - currErr) / baseErr;
                    const pct = (improvement * 100).toFixed(1);
                    const cls = improvement > 0 ? 'up' : improvement < 0 ? 'down' : '';
                    const sign = improvement > 0 ? '+' : '';
                    delta = `<span class="delta ${cls}">${sign}${pct}%</span>`;
                }
            }
            return `<tr><td>${key}</td><td>${cell}${delta}</td></tr>`;
        }).join('');
        return `<table class="prop-table"><tbody>${rows}</tbody></table>`;
    };

    const renderMoleculeCard = (container, payload, { baseline = null } = {}) => {
        if (!payload) return;
        container.classList.remove('empty');
        const mape = payload.mape;
        const badge = mape !== undefined && mape !== null
            ? `<span class="badge ${mapeBadgeClass(mape)}">MAPE ${fmt(mape, 1)}%</span>`
            : '';
        container.innerHTML = `
            ${payload.image ? `<img src="${payload.image}" class="mol-image" alt="Molecule structure">` : ''}
            <div class="mol-smiles" title="${payload.smiles}">${payload.smiles}</div>
            ${molPropertyTable(payload.properties, { target: state.targetProperties, baseline })}
            <div class="candidate-meta">
                ${badge}
                <span>Feas. ${((1 - payload.feasibility) * 100).toFixed(0)}%</span>
            </div>
        `;
    };

    const renderTarget = (props) => {
        els.target.classList.remove('empty');
        const rows = PROP_KEYS.map((k) => `<tr><td>${k}</td><td>${fmtProp(k, props[k])}</td></tr>`).join('');
        els.target.innerHTML = `<table class="prop-table"><tbody>${rows}</tbody></table>`;
    };

    const renderCandidates = (iteration, candidates) => {
        if (!candidates || candidates.length === 0) {
            els.candidates.innerHTML = '<div class="empty-panel">No feasible candidates in this iteration.</div>';
            return;
        }
        els.candSub.textContent = `Iteration ${iteration} · ${candidates.length} shown`;
        const html = candidates.map((c, i) => {
            const mape = c.mape;
            const badge = `<span class="badge ${mapeBadgeClass(mape)}">MAPE ${fmt(mape, 1)}%</span>`;
            return `
                <button type="button" class="candidate-card" data-idx="${i}">
                    ${c.image ? `<img src="${c.image}" class="mol-image" alt="">` : ''}
                    <div class="mol-smiles" title="${c.smiles}">${c.smiles}</div>
                    ${molPropertyTable(c.properties, { target: state.targetProperties })}
                    <div class="candidate-meta">
                        ${badge}
                        <span>Feas. ${((1 - c.feasibility) * 100).toFixed(0)}%</span>
                    </div>
                </button>
            `;
        }).join('');
        els.candidates.innerHTML = html;
        // Attach drawer handlers
        els.candidates.querySelectorAll('.candidate-card').forEach((node, idx) => {
            node.addEventListener('click', () => openDrawer(candidates[idx]));
        });
    };

    // ---------------------------------------------------------------- drawer
    const openDrawer = (payload) => {
        els.drawerBody.innerHTML = `
            ${payload.image ? `<img src="${payload.image}" class="mol-image" alt="">` : ''}
            <div class="mol-smiles">${payload.smiles}</div>
            ${molPropertyTable(payload.properties, { target: state.targetProperties })}
            <div class="candidate-meta" style="margin-top:12px">
                <span class="badge ${mapeBadgeClass(payload.mape)}">MAPE ${fmt(payload.mape, 2)}%</span>
                <span>Feasibility ${((1 - payload.feasibility) * 100).toFixed(0)}%</span>
            </div>
            <p style="font-size:12px;color:var(--text-muted);margin-top:14px">
                Combined score: ${fmt(payload.score, 4)}
            </p>
        `;
        els.drawer.setAttribute('aria-hidden', 'false');
    };

    const closeDrawer = () => els.drawer.setAttribute('aria-hidden', 'true');
    els.drawerClose.addEventListener('click', closeDrawer);
    els.drawer.addEventListener('click', (e) => {
        if (e.target === els.drawer) closeDrawer();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDrawer();
    });

    // ---------------------------------------------------------------- actions
    const setRunning = (running) => {
        state.running = running;
        els.runBtn.disabled = running;
        els.stopBtn.disabled = !running;
        els.runPill.dataset.state = running ? 'running' : 'idle';
        els.runPill.textContent = running ? 'Running' : 'Idle';
        if (running) {
            els.progressBar.classList.add('indeterminate');
        } else {
            els.progressBar.classList.remove('indeterminate');
        }
    };

    const setStatus = (msg, cls = '') => {
        els.statusMsg.className = `status-line ${cls}`;
        els.statusMsg.textContent = msg || '';
    };

    const start = async () => {
        const payload = {
            density: parseFloat(els.density.value),
            velocity: parseFloat(els.velocity.value),
            pressure: parseFloat(els.pressure.value),
            hf: parseFloat(els.hf.value),
            beam_width: parseInt(els.beamWidth.value, 10),
            top_k: parseInt(els.topK.value, 10),
            max_iter: parseInt(els.maxIter.value, 10),
            mape_threshold: parseFloat(els.mape.value),
            enable_rag: els.enableRag.checked,
        };

        state.maxIter = payload.max_iter;
        state.iteration = 0;
        els.iterCount.textContent = '0';
        els.candCount.textContent = '0';
        els.bestMape.textContent = '–';
        els.bestFeas.textContent = '–';
        els.bestScore.textContent = '–';
        els.progressFill.style.width = '0%';
        els.progressSub.textContent = 'Starting…';
        els.candidates.innerHTML = '<div class="empty-panel">Searching…</div>';
        els.candSub.textContent = 'Running';
        els.seedMol.classList.add('empty');
        els.seedMol.innerHTML = '<p class="empty-text">Loading…</p>';
        els.bestMol.classList.add('empty');
        els.bestMol.innerHTML = '<p class="empty-text">No results yet</p>';
        state.seedProperties = null;

        try {
            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const body = await res.json().catch(() => ({}));
                setStatus(body.message || `Start failed (${res.status}).`, 'error');
                return;
            }
        } catch (e) {
            setStatus(`Network error: ${e}`, 'error');
            return;
        }

        setRunning(true);
        setStatus('Starting optimization…');
        connectStream();
    };

    const stop = async () => {
        try {
            await fetch('/api/stop', { method: 'POST' });
            setStatus('Stop requested…');
        } catch (e) {
            setStatus(`Stop failed: ${e}`, 'error');
        }
    };

    const connectStream = () => {
        if (state.es) state.es.close();
        state.es = new EventSource('/api/progress');
        state.es.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                handleEvent(data);
            } catch (_) {
                /* ignore */
            }
        };
        state.es.onerror = () => {
            // Let SSE auto-reconnect; no action.
        };
    };

    const handleEvent = (data) => {
        switch (data.type) {
            case 'heartbeat':
                return;
            case 'status':
                setStatus(data.message);
                els.progressSub.textContent = data.message;
                return;
            case 'target':
                state.targetProperties = data.properties;
                renderTarget(data.properties);
                return;
            case 'seed':
                state.seedProperties = data.properties;
                renderMoleculeCard(els.seedMol, data);
                els.compSub.textContent = 'Seed loaded';
                return;
            case 'iteration':
                state.iteration = data.iteration;
                els.iterCount.textContent = String(data.iteration);
                els.candCount.textContent = String(data.candidates.length);
                els.progressFill.style.width =
                    `${Math.min(100, (data.iteration / state.maxIter) * 100)}%`;
                els.progressSub.textContent =
                    `Iteration ${data.iteration} of up to ${state.maxIter}`;
                // Stats based on best candidate
                if (data.candidates.length > 0) {
                    const best = data.candidates[0];
                    els.bestMape.textContent = `${fmt(best.mape, 2)}%`;
                    els.bestFeas.textContent = `${((1 - best.feasibility) * 100).toFixed(0)}%`;
                    els.bestScore.textContent = fmt(best.score, 3);
                }
                renderCandidates(data.iteration, data.candidates);
                return;
            case 'best':
                renderMoleculeCard(els.bestMol, data, { baseline: state.seedProperties });
                els.compSub.textContent = `Best found at iteration ${state.iteration}`;
                return;
            case 'complete':
                setRunning(false);
                setStatus('Optimization complete.');
                els.runPill.dataset.state = 'done';
                els.runPill.textContent = 'Done';
                els.progressFill.style.width = '100%';
                if (state.es) state.es.close();
                return;
            case 'error':
                setRunning(false);
                setStatus(data.message || 'Error', 'error');
                els.runPill.dataset.state = 'error';
                els.runPill.textContent = 'Error';
                if (state.es) state.es.close();
                return;
            default:
                return;
        }
    };

    // ---------------------------------------------------------------- init
    els.runBtn.addEventListener('click', start);
    els.stopBtn.addEventListener('click', stop);
    els.enableRag.addEventListener('change', () => {
        els.ragPill.dataset.state = els.enableRag.checked ? 'on' : 'off';
        els.ragPill.innerHTML = `RAG <strong>${els.enableRag.checked ? 'On' : 'Off'}</strong>`;
    });
})();
