"""
Flask web application for the Energetic Molecular Design System.

This module no longer re-implements the beam search loop — it registers
observer callbacks on :class:`BeamSearchEngine` and streams progress to
the browser via Server-Sent Events.
"""

import base64
import io
import json
import logging
import os
import sys
import threading
import time
from queue import Queue
from typing import List, Optional

from flask import Flask, Response, jsonify, render_template, request, make_response

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dotenv import load_dotenv
load_dotenv()

from data_structures import PropertyTarget, MoleculeState  # noqa: E402
from config import Config  # noqa: E402
from designer import EnergeticDesigner  # noqa: E402
from orchestrator import BeamSearchEngine  # noqa: E402


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def _apply_cors_headers(response):
    """Allow browser UIs on other origins (e.g. GitHub Pages) when EMG_CORS_ORIGINS is set.

    Example:  EMG_CORS_ORIGINS=https://yourname.github.io
    or:       EMG_CORS_ORIGINS=*
    """
    allow = (os.getenv('EMG_CORS_ORIGINS') or '').strip()
    if not allow:
        return response
    origin = request.headers.get('Origin', '')
    if allow == '*':
        response.headers['Access-Control-Allow-Origin'] = '*'
    else:
        allowed = {x.strip() for x in allow.split(',') if x.strip()}
        if origin in allowed:
            response.headers['Access-Control-Allow-Origin'] = origin
    if 'Access-Control-Allow-Origin' in response.headers:
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


@app.after_request
def _cors_after(response):
    if request.path.startswith('/api'):
        return _apply_cors_headers(response)
    return response


@app.before_request
def _cors_preflight():
    if request.method != 'OPTIONS' or not request.path.startswith('/api'):
        return None
    if not (os.getenv('EMG_CORS_ORIGINS') or '').strip():
        return None
    r = make_response('', 204)
    return _apply_cors_headers(r)


progress_queue: "Queue[dict]" = Queue()
_current_engine: Optional[BeamSearchEngine] = None
_current_thread: Optional[threading.Thread] = None


def generate_molecule_image(smiles: str, size=(200, 200)) -> Optional[str]:
    """Render a SMILES string to a base64 PNG data URL."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buffer.read()).decode("utf-8")}'
    except Exception as e:
        print(f"Error generating molecule image: {e}")
        return None


def _molecule_payload(mol: MoleculeState, size=(220, 220), mape: Optional[float] = None) -> dict:
    sources = getattr(mol, 'property_sources', {}) or {}
    citations = getattr(mol, 'citations', []) or []
    # Compact per-property origin flag for the UI: 'lit' | 'pred' | 'data'.
    origin = {}
    for k, v in sources.items():
        s = (v or '').lower()
        if 'analogue' in s:
            origin[k] = 'ana'
        elif 'literature' in s:
            origin[k] = 'lit'
        elif 'dataset' in s:
            origin[k] = 'data'
        else:
            origin[k] = 'pred'
    return {
        'smiles': mol.smiles,
        'properties': mol.properties,
        'feasibility': mol.feasibility,
        'score': mol.score,
        'mape': mape,
        'image': generate_molecule_image(mol.smiles, size=size),
        'property_origin': origin,
        'property_sources': sources,
        'citations': citations,
        'lit_hits': sum(1 for o in origin.values() if o in ('lit', 'ana')),
    }


def run_beam_search(target_props, enable_rag, use_llm,
                    ollama_base_url, ollama_model,
                    beam_width, top_k, max_iter, mape_threshold_pct):
    """Background worker that drives :class:`BeamSearchEngine`."""
    global _current_engine

    try:
        while not progress_queue.empty():
            progress_queue.get()

        target = PropertyTarget(
            density=target_props['density'],
            det_velocity=target_props['velocity'],
            det_pressure=target_props['pressure'],
            hf_solid=target_props['hf'],
        )

        config = Config()
        config.beam_search.beam_width = beam_width
        config.beam_search.top_k = top_k
        config.beam_search.max_iterations = max_iter
        # UI "MAPE threshold" = absolute target: stop when best MAPE ≤ this.
        config.beam_search.mape_target = float(mape_threshold_pct)
        config.literature.enable_literature_search = bool(enable_rag)
        config.literature.use_llm = bool(use_llm)
        config.literature.ollama_base_url = ollama_base_url or None
        config.literature.ollama_model = ollama_model or 'ALIENTELLIGENCE/chemicalengineer'

        parent_dir = os.path.join(os.path.dirname(__file__), '..')
        config.system.dataset_path = os.path.join(parent_dir, 'sample_start_molecules.csv')
        config.system.models_directory = os.path.join(parent_dir, 'models')
        config.system.output_directory = os.path.join(parent_dir, 'output')
        config.literature.cache_path = os.path.join(config.system.output_directory, 'literature_cache.sqlite')

        designer = EnergeticDesigner(target, config)

        progress_queue.put({'type': 'status', 'message': 'Initializing system…'})
        designer.initialize()

        progress_queue.put({
            'type': 'target',
            'properties': target.to_dict(),
        })

        engine = BeamSearchEngine(config, target)
        _current_engine = engine

        if engine.literature_retriever is not None and engine.literature_retriever.cache is not None:
            engine.literature_retriever.cache.clear()
            engine.literature_retriever._analogue_mem.clear()

        has_openai = bool(config.literature.openai_api_key)
        has_ollama = bool(config.literature.ollama_base_url)
        llm_available = has_openai or has_ollama
        backend = ('ollama' if has_ollama else 'openai') if llm_available else 'none'
        progress_queue.put({
            'type': 'literature_status',
            'enabled': bool(config.literature.enable_literature_search and engine.literature_retriever is not None),
            'use_llm': bool(config.literature.use_llm and llm_available),
            'llm_analogue': llm_available,
            'llm_backend': backend,
            'ollama_model': config.literature.ollama_model if has_ollama else '',
            'cache_path': config.literature.cache_path,
        })

        # --- Callback wiring ----------------------------------------------

        def on_seed(mol: MoleculeState):
            progress_queue.put({
                'type': 'seed',
                **_molecule_payload(mol, size=(280, 280),
                                    mape=engine.calculate_mape(mol)),
            })

        def on_iteration(iteration: int,
                         all_candidates: List[MoleculeState],
                         beam: List[MoleculeState]):
            feasible = [c for c in all_candidates if c.is_feasible]
            # Deduplicate by SMILES and order by MAPE ascending for display.
            uniq: dict = {}
            for c in feasible:
                uniq.setdefault(c.smiles, c)
            display = sorted(uniq.values(), key=engine.calculate_mape)[:beam_width]
            payload = [_molecule_payload(c, size=(180, 180),
                                         mape=engine.calculate_mape(c))
                       for c in display]
            progress_queue.put({
                'type': 'iteration',
                'iteration': iteration,
                'candidates': payload,
                'beam_size': len(beam),
                'top_k': top_k,
            })

        def on_best(mol: MoleculeState):
            progress_queue.put({
                'type': 'best',
                **_molecule_payload(mol, size=(280, 280),
                                    mape=engine.calculate_mape(mol)),
            })

        def on_status(msg: str):
            progress_queue.put({'type': 'status', 'message': msg})

        def on_complete(_best: MoleculeState):
            progress_queue.put({'type': 'complete'})

        engine.on_seed = on_seed
        engine.on_iteration = on_iteration
        engine.on_best = on_best
        engine.on_status = on_status
        engine.on_complete = on_complete

        progress_queue.put({'type': 'status', 'message': 'Running beam search…'})
        engine.run(designer.seed)

    except Exception as e:
        import traceback
        progress_queue.put({'type': 'error', 'message': str(e)})
        print(traceback.format_exc())
    finally:
        _current_engine = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_search():
    global _current_thread

    if _current_thread is not None and _current_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'search already running'}), 409

    data = request.json or {}
    target_props = {
        'density': float(data.get('density', 1.5)),
        'velocity': float(data.get('velocity', 7000)),
        'pressure': float(data.get('pressure', 20)),
        'hf': float(data.get('hf', 100)),
    }
    enable_rag = bool(data.get('enable_rag', True))
    use_llm = bool(data.get('use_llm', False))
    ollama_base_url = str(data.get('ollama_base_url', '') or '').strip()
    if ollama_base_url and not ollama_base_url.startswith(('http://', 'https://')):
        ollama_base_url = 'http://' + ollama_base_url
    ollama_base_url = ollama_base_url or None
    ollama_model = str(data.get('ollama_model', '') or '').strip() or 'ALIENTELLIGENCE/chemicalengineer'
    beam_width = int(data.get('beam_width', 10))
    top_k = int(data.get('top_k', 5))
    max_iter = int(data.get('max_iter', 10))
    mape_threshold_pct = float(data.get('mape_threshold', 1.0))

    thread = threading.Thread(
        target=run_beam_search,
        args=(target_props, enable_rag, use_llm,
              ollama_base_url, ollama_model,
              beam_width, top_k, max_iter, mape_threshold_pct),
        daemon=True,
    )
    thread.start()
    _current_thread = thread

    return jsonify({'status': 'started'})


@app.route('/api/key-status')
def key_status():
    """Report LLM backend availability without exposing secrets."""
    from config import Config
    cfg = Config()
    key = cfg.literature.openai_api_key or os.getenv('OPENAI_API_KEY') or ''
    has_key = bool(key.strip())
    hint = (key[:4] + '…') if has_key else ''

    ollama_url = cfg.literature.ollama_base_url or os.getenv('OLLAMA_BASE_URL') or ''
    ollama_model = cfg.literature.ollama_model or os.getenv('OLLAMA_MODEL') or 'ALIENTELLIGENCE/chemicalengineer'

    # Quick reachability probe for Ollama (non-blocking, short timeout).
    ollama_reachable = False
    if ollama_url:
        try:
            import urllib.request
            normalized = ollama_url.strip().rstrip('/')
            if not normalized.startswith(('http://', 'https://')):
                normalized = 'http://' + normalized
            probe = normalized + '/api/tags'
            req = urllib.request.urlopen(probe, timeout=2)
            ollama_reachable = req.status == 200
        except Exception:
            ollama_reachable = False

    return jsonify({
        'has_key': has_key,
        'hint': hint,
        'ollama_url': ollama_url,
        'ollama_model': ollama_model,
        'ollama_reachable': ollama_reachable,
    })


@app.route('/api/stop', methods=['POST'])
def stop_search():
    if _current_engine is not None:
        _current_engine.request_stop()
        return jsonify({'status': 'stopping'})
    return jsonify({'status': 'idle'})


@app.route('/api/progress')
def progress_stream():
    def generate():
        while True:
            if not progress_queue.empty():
                update = progress_queue.get()
                yield f"data: {json.dumps(update)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            time.sleep(0.1)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    print()
    print("  Energetic Molecular Designer — GUI server")
    print("  → http://localhost:5001")
    print("  (Ctrl+C to stop)")
    print()
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5001)
