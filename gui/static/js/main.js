/* Energetic Designer — vanilla JS front-end for the Flask SSE backend. */

(() => {
    const $ = (id) => document.getElementById(id);

    const state = {
        es: null,
        running: false,
        iteration: 0,
        maxIter: 10,
        targetProperties: null,
        lit: {
            enabled: false,
            useLlm: false,
            moleculesQueried: 0,
            literatureHits: 0,
            papers: new Map(),
        },
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
        enableLit: $('enable-rag'),
        useLlm: $('use-llm'),
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
        litPill: $('lit-pill'),
        runPill: $('run-pill'),
        litSub: $('lit-sub'),
        litState: $('lit-state'),
        litMolecules: $('lit-molecules'),
        litHits: $('lit-hits'),
        litPapers: $('lit-papers'),
        litCitations: $('lit-citations'),
        apiKeyStatus: $('api-key-status'),
        ollamaStatus: $('ollama-status'),
        ollamaUrl: $('ollama-url'),
        ollamaModel: $('ollama-model'),
        tabOpenai: $('tab-openai'),
        tabOllama: $('tab-ollama'),
        panelOpenai: $('panel-openai'),
        panelOllama: $('panel-ollama'),
        llmBackendSection: $('llm-backend-section'),
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

    const ORIGIN_LABEL = { lit: 'lit', ana: 'ana', pred: 'pred', data: 'data' };
    const ORIGIN_TITLE = {
        lit: 'from literature search',
        ana: 'from literature analogue (fallback)',
        pred: 'predicted (XGBoost)',
        data: 'from dataset',
    };

    const originChip = (origin) => {
        if (!origin) return '';
        const cls = origin in ORIGIN_LABEL ? origin : 'pred';
        return `<span class="origin ${cls}" title="${ORIGIN_TITLE[cls]}">${ORIGIN_LABEL[cls]}</span>`;
    };

    const molPropertyTable = (props, { target = null, origin = null } = {}) => {
        const rows = PROP_KEYS.map((key) => {
            const cell = fmtProp(key, props?.[key]);
            let delta = '';
            if (target && props?.[key] !== undefined && target[key] !== undefined) {
                const tgt = target[key];
                if (Math.abs(tgt) > 0) {
                    const pct = ((props[key] - tgt) / Math.abs(tgt)) * 100;
                    const abs = Math.abs(pct);
                    const cls = abs < 2 ? 'up' : abs < 10 ? '' : 'down';
                    const sign = pct > 0 ? '+' : '';
                    delta = `<span class="delta ${cls}" title="vs. target">${sign}${pct.toFixed(1)}%</span>`;
                }
            }
            const chip = origin && origin[key] ? originChip(origin[key]) : '';
            return `<tr><td>${key}${chip}</td><td>${cell}${delta}</td></tr>`;
        }).join('');
        return `<table class="prop-table"><tbody>${rows}</tbody></table>`;
    };

    const escapeHtml = (s) => String(s || '').replace(/[&<>"']/g,
        (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    const doiLink = (doi) => {
        if (!doi) return '';
        const clean = doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '');
        return `<a href="https://doi.org/${encodeURI(clean)}" target="_blank" rel="noopener">${escapeHtml(clean)}</a>`;
    };

    const renderCitationItem = (c) => {
        const title = escapeHtml(c.title || 'Untitled');
        const authors = (c.authors || []).slice(0, 3).map(escapeHtml).join(', ');
        const more = (c.authors || []).length > 3 ? ' et al.' : '';
        const source = escapeHtml(c.source_db || '');
        const doi = doiLink(c.doi);
        const props = (c.properties_found || []).map((p) =>
            `<span>${escapeHtml(p)}</span>`).join('');
        return `
            <li>
                <span class="citation-title">${title}</span>
                <span class="citation-meta">
                    ${authors}${more}
                    ${source ? ` · ${source}` : ''}
                    ${doi ? ` · ${doi}` : ''}
                </span>
                ${props ? `<div class="citation-props">${props}</div>` : ''}
            </li>
        `;
    };

    const renderInlineCitations = (citations) => {
        if (!citations || citations.length === 0) return '';
        return `<ul class="citation-list">${citations.map(renderCitationItem).join('')}</ul>`;
    };

    const litChip = (hits) => {
        if (!hits || hits <= 0) return '';
        return `<span class="lit-chip" title="Properties sourced from literature">Lit ${hits}/4</span>`;
    };

    const renderMoleculeCard = (container, payload) => {
        if (!payload) return;
        container.classList.remove('empty');
        const mape = payload.mape;
        const badge = mape !== undefined && mape !== null
            ? `<span class="badge ${mapeBadgeClass(mape)}">MAPE ${fmt(mape, 1)}%</span>`
            : '';
        container.innerHTML = `
            ${payload.image ? `<img src="${payload.image}" class="mol-image" alt="Molecule structure">` : ''}
            <div class="mol-smiles" title="${payload.smiles}">${payload.smiles}</div>
            ${molPropertyTable(payload.properties, {
                target: state.targetProperties,
                origin: payload.property_origin,
            })}
            <div class="candidate-meta">
                ${badge}
                ${litChip(payload.lit_hits)}
                <span>Feas. ${((1 - payload.feasibility) * 100).toFixed(0)}%</span>
            </div>
            ${renderInlineCitations(payload.citations)}
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
                    ${molPropertyTable(c.properties, {
                        target: state.targetProperties,
                        origin: c.property_origin,
                    })}
                    <div class="candidate-meta">
                        ${badge}
                        ${litChip(c.lit_hits)}
                        <span>Feas. ${((1 - c.feasibility) * 100).toFixed(0)}%</span>
                    </div>
                </button>
            `;
        }).join('');
        els.candidates.innerHTML = html;
        els.candidates.querySelectorAll('.candidate-card').forEach((node, idx) => {
            node.addEventListener('click', () => openDrawer(candidates[idx]));
        });
    };

    // ---------------------------------------------------------------- drawer
    const openDrawer = (payload) => {
        els.drawerBody.innerHTML = `
            ${payload.image ? `<img src="${payload.image}" class="mol-image" alt="">` : ''}
            <div class="mol-smiles">${payload.smiles}</div>
            ${molPropertyTable(payload.properties, {
                target: state.targetProperties,
                origin: payload.property_origin,
            })}
            <div class="candidate-meta" style="margin-top:12px">
                <span class="badge ${mapeBadgeClass(payload.mape)}">MAPE ${fmt(payload.mape, 2)}%</span>
                ${litChip(payload.lit_hits)}
                <span>Feasibility ${((1 - payload.feasibility) * 100).toFixed(0)}%</span>
            </div>
            <p style="font-size:12px;color:var(--text-muted);margin-top:14px">
                Combined score: ${fmt(payload.score, 4)}
            </p>
            ${renderInlineCitations(payload.citations)}
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
            enable_rag: els.enableLit.checked,
            use_llm: els.enableLit.checked && els.useLlm.checked,
            ollama_base_url: _activeBackend === 'ollama'
                ? (els.ollamaUrl.value || '').trim() : '',
            ollama_model: (els.ollamaModel.value || 'llama3.2').trim(),
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
        setLitState(els.enableLit.checked, els.enableLit.checked && els.useLlm.checked, false);
        resetLit();

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
        state.es.onerror = () => {};
    };

    const ingestLitPayload = (payload) => {
        if (!state.lit.enabled || !payload) return;
        state.lit.moleculesQueried += 1;
        if (payload.lit_hits) state.lit.literatureHits += payload.lit_hits;
        for (const c of (payload.citations || [])) {
            const key = (c.doi && c.doi.trim()) || c.title || '';
            if (!key) continue;
            const prev = state.lit.papers.get(key);
            if (prev) {
                const merged = new Set([
                    ...(prev.properties_found || []),
                    ...(c.properties_found || []),
                ]);
                prev.properties_found = [...merged];
            } else {
                state.lit.papers.set(key, { ...c });
            }
        }
        renderLitCard();
    };

    const renderLitCard = () => {
        els.litMolecules.textContent = String(state.lit.moleculesQueried);
        els.litHits.textContent = String(state.lit.literatureHits);
        els.litPapers.textContent = String(state.lit.papers.size);
        if (state.lit.papers.size === 0) {
            els.litCitations.innerHTML = `<li class="empty-text">${
                state.lit.enabled
                    ? 'No literature hits yet.'
                    : 'Enable literature search to retrieve papers.'
            }</li>`;
            return;
        }
        const items = [...state.lit.papers.values()]
            .sort((a, b) => (b.properties_found?.length || 0) - (a.properties_found?.length || 0))
            .slice(0, 20);
        els.litCitations.innerHTML = items.map(renderCitationItem).join('');
    };

    const setLitState = (enabled, useLlm, llmAnalogue) => {
        state.lit.enabled = !!enabled;
        state.lit.useLlm = !!useLlm;
        els.litPill.dataset.state = enabled ? 'on' : 'off';
        els.litPill.innerHTML = `Lit <strong>${enabled ? 'On' : 'Off'}</strong>`;
        let mode = 'regex';
        if (useLlm && llmAnalogue) mode = 'LLM extract + analogue';
        else if (useLlm) mode = 'LLM extract';
        else if (llmAnalogue) mode = 'regex · LLM analogue';
        els.litState.textContent = enabled ? `On · ${mode}` : 'Off';
        els.litSub.textContent = enabled
            ? `Literature search · ${mode}`
            : 'Disabled — all properties predicted';
    };

    const resetLit = () => {
        state.lit.moleculesQueried = 0;
        state.lit.literatureHits = 0;
        state.lit.papers = new Map();
        renderLitCard();
    };

    const handleEvent = (data) => {
        switch (data.type) {
            case 'heartbeat':
                return;
            case 'status':
                setStatus(data.message);
                els.progressSub.textContent = data.message;
                return;
            case 'literature_status': {
                setLitState(data.enabled, data.use_llm, data.llm_analogue);
                if (data.llm_backend === 'ollama') {
                    setOllamaStatus(true, els.ollamaUrl.value, data.ollama_model || '');
                } else {
                    setApiKeyStatus(data.llm_analogue, '');
                }
                resetLit();
                return;
            }
            case 'target':
                state.targetProperties = data.properties;
                renderTarget(data.properties);
                return;
            case 'seed':
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
                if (data.candidates.length > 0) {
                    const best = data.candidates[0];
                    els.bestMape.textContent = `${fmt(best.mape, 2)}%`;
                    els.bestFeas.textContent = `${((1 - best.feasibility) * 100).toFixed(0)}%`;
                    els.bestScore.textContent = fmt(best.score, 3);
                }
                for (const c of (data.candidates || [])) ingestLitPayload(c);
                renderCandidates(data.iteration, data.candidates);
                return;
            case 'best':
                renderMoleculeCard(els.bestMol, data);
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
    let _activeBackend = 'openai';

    const setOllamaStatus = (reachable, url, model) => {
        const el = els.ollamaStatus;
        if (!url) {
            el.dataset.state = '';
            el.querySelector('.api-key-label').textContent = 'Enter server URL to check';
            return;
        }
        if (reachable) {
            el.dataset.state = 'present';
            el.querySelector('.api-key-label').textContent =
                `Reachable · model: ${model || 'llama3.2'}`;
        } else {
            el.dataset.state = 'missing';
            el.querySelector('.api-key-label').textContent =
                `Cannot reach ${url}`;
        }
    };

    const setApiKeyStatus = (hasKey, hint) => {
        const el = els.apiKeyStatus;
        if (hasKey) {
            el.dataset.state = 'present';
            el.querySelector('.api-key-label').textContent =
                `API key present${hint ? ` (${hint})` : ''}`;
        } else {
            el.dataset.state = 'missing';
            el.querySelector('.api-key-label').textContent =
                'No API key — set OPENAI_API_KEY in .env';
        }
        syncLlmToggle();
    };

    const probeOllama = () => {
        const url = (els.ollamaUrl.value || '').trim();
        if (!url) { setOllamaStatus(false, '', ''); return; }
        fetch('/api/key-status')
            .then((r) => r.json())
            .then((d) => setOllamaStatus(d.ollama_reachable, d.ollama_url || url, d.ollama_model))
            .catch(() => setOllamaStatus(false, url, ''));
    };

    const switchTab = (backend) => {
        _activeBackend = backend;
        els.tabOpenai.classList.toggle('active', backend === 'openai');
        els.tabOllama.classList.toggle('active', backend === 'ollama');
        els.panelOpenai.hidden = (backend !== 'openai');
        els.panelOllama.hidden = (backend !== 'ollama');
    };

    els.tabOpenai.addEventListener('click', () => switchTab('openai'));
    els.tabOllama.addEventListener('click', () => switchTab('ollama'));
    els.ollamaUrl.addEventListener('change', probeOllama);
    els.ollamaUrl.addEventListener('blur', probeOllama);

    fetch('/api/key-status')
        .then((r) => r.json())
        .then((d) => {
            setApiKeyStatus(d.has_key, d.hint);
            if (d.ollama_url) {
                els.ollamaUrl.value = d.ollama_url;
                els.ollamaModel.value = d.ollama_model || 'llama3.2';
                setOllamaStatus(d.ollama_reachable, d.ollama_url, d.ollama_model);
                if (d.ollama_reachable) switchTab('ollama');
            }
        })
        .catch(() => setApiKeyStatus(false, ''));

    els.runBtn.addEventListener('click', start);
    els.stopBtn.addEventListener('click', stop);

    const syncLlmToggle = () => {
        const litOn = els.enableLit.checked;
        els.useLlm.disabled = !litOn;
        if (!litOn) els.useLlm.checked = false;
        els.useLlm.closest('label').style.opacity = litOn ? '' : '0.5';
        els.llmBackendSection.style.opacity = litOn ? '' : '0.5';
        setLitState(litOn, litOn && els.useLlm.checked, false);
    };

    els.enableLit.addEventListener('change', syncLlmToggle);
    els.useLlm.addEventListener('change', syncLlmToggle);

    syncLlmToggle();
    renderLitCard();
})();
