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

from flask import Flask, Response, jsonify, render_template, request

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
    return {
        'smiles': mol.smiles,
        'properties': mol.properties,
        'feasibility': mol.feasibility,
        'score': mol.score,
        'mape': mape,
        'image': generate_molecule_image(mol.smiles, size=size),
    }


def run_beam_search(target_props, enable_rag, beam_width, top_k, max_iter, mape_threshold_pct):
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
        # Convergence threshold (in MAPE %) — the orchestrator compares
        # MAPE-to-MAPE now, so this is directly comparable.
        config.beam_search.convergence_threshold = float(mape_threshold_pct)
        config.rag.enable_rag = bool(enable_rag)

        parent_dir = os.path.join(os.path.dirname(__file__), '..')
        config.system.dataset_path = os.path.join(parent_dir, 'sample_start_molecules.csv')
        config.system.models_directory = os.path.join(parent_dir, 'models')
        config.system.output_directory = os.path.join(parent_dir, 'output')
        config.rag.cache_path = os.path.join(config.system.output_directory, 'rag_cache.sqlite')

        designer = EnergeticDesigner(target, config)

        progress_queue.put({'type': 'status', 'message': 'Initializing system…'})
        designer.initialize()

        progress_queue.put({
            'type': 'target',
            'properties': target.to_dict(),
        })

        engine = BeamSearchEngine(config, target)
        _current_engine = engine

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
    beam_width = int(data.get('beam_width', 10))
    top_k = int(data.get('top_k', 5))
    max_iter = int(data.get('max_iter', 10))
    mape_threshold_pct = float(data.get('mape_threshold', 1.0))

    thread = threading.Thread(
        target=run_beam_search,
        args=(target_props, enable_rag, beam_width, top_k, max_iter, mape_threshold_pct),
        daemon=True,
    )
    thread.start()
    _current_thread = thread

    return jsonify({'status': 'started'})


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
