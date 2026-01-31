"""
Flask web application for Molecular Design System GUI.
Provides a modern web interface for beam search optimization.
"""

import os
import sys
import json
import time
import logging
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from queue import Queue
import threading

# Suppress RDKit warnings (must be done before importing other modules that use RDKit)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Suppress verbose logging - only show warnings and errors
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Suppress Flask's werkzeug logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_structures import PropertyTarget
from config import Config
from designer import EnergeticDesigner

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Global state for progress updates
progress_queue = Queue()
current_search = None


def generate_molecule_image(smiles: str, filename: str, size=(200, 200)):
    """Generate molecule image from SMILES as base64 data URL (no file save)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        import base64
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate image in memory
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64 data URL
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return f'data:image/png;base64,{img_base64}'
        
    except Exception as e:
        print(f"Error generating molecule image: {e}")
        return None


class ProgressReporter:
    """Reports beam search progress to the web interface."""
    
    def __init__(self, queue):
        self.queue = queue
    
    def report_seed(self, molecule):
        """Report initial seed molecule."""
        img_path = generate_molecule_image(molecule.smiles, 'seed.png', size=(300, 300))
        self.queue.put({
            'type': 'seed',
            'smiles': molecule.smiles,
            'properties': molecule.properties,
            'feasibility': molecule.feasibility,
            'score': molecule.score,
            'image': img_path
        })
    
    def report_iteration(self, iteration, candidates, beam, top_k, mape_calculator=None):
        """Report iteration progress."""
        # Generate images for all candidates (already limited to beam_width by caller)
        candidate_data = []
        for i, candidate in enumerate(candidates):
            img_path = generate_molecule_image(
                candidate.smiles, 
                f'iter_{iteration}_cand_{i}.png',
                size=(150, 150)
            )
            # Calculate MAPE if calculator provided
            mape = mape_calculator(candidate) if mape_calculator else 0
            candidate_data.append({
                'smiles': candidate.smiles,
                'properties': candidate.properties,
                'feasibility': candidate.feasibility,
                'score': candidate.score,
                'mape': mape,
                'image': img_path
            })
        
        self.queue.put({
            'type': 'iteration',
            'iteration': iteration,
            'candidates': candidate_data,
            'beam_size': len(beam),
            'top_k': top_k
        })
    
    def report_best(self, molecule):
        """Report best molecule found."""
        img_path = generate_molecule_image(molecule.smiles, 'best.png', size=(300, 300))
        self.queue.put({
            'type': 'best',
            'smiles': molecule.smiles,
            'properties': molecule.properties,
            'feasibility': molecule.feasibility,
            'score': molecule.score,
            'image': img_path
        })
    
    def report_complete(self):
        """Report search completion."""
        self.queue.put({'type': 'complete'})


def run_beam_search(target_props, enable_rag, beam_width, top_k, max_iter, mape_threshold_pct):
    """Run beam search in background thread."""
    global current_search, progress_queue
    
    try:
        # Clear queue
        while not progress_queue.empty():
            progress_queue.get()
        
        # Create target
        target = PropertyTarget(
            density=target_props['density'],
            det_velocity=target_props['velocity'],
            det_pressure=target_props['pressure'],
            hf_solid=target_props['hf']
        )
        
        # Configure system with correct paths (parent directory)
        config = Config()
        config.beam_search.beam_width = beam_width
        config.beam_search.top_k = top_k
        config.beam_search.max_iterations = max_iter
        config.rag.enable_rag = enable_rag
        
        # Update paths to parent directory
        parent_dir = os.path.join(os.path.dirname(__file__), '..')
        config.system.dataset_path = os.path.join(parent_dir, 'sample_start_molecules.csv')
        config.system.models_directory = os.path.join(parent_dir, 'models')
        config.system.output_directory = os.path.join(parent_dir, 'output')
        config.rag.chroma_persist_directory = os.path.join(parent_dir, 'chroma_db')
        
        # Create designer
        designer = EnergeticDesigner(target, config)
        reporter = ProgressReporter(progress_queue)
        
        # Initialize
        progress_queue.put({'type': 'status', 'message': 'Initializing system...'})
        designer.initialize()
        
        # Report seed
        reporter.report_seed(designer.seed)
        
        # Report target
        progress_queue.put({
            'type': 'target',
            'properties': target.to_dict()
        })
        
        # Run with progress reporting
        progress_queue.put({'type': 'status', 'message': 'Running beam search...'})
        
        # Modified run loop with reporting
        from orchestrator import BeamSearchEngine
        engine = BeamSearchEngine(config, target)
        
        beam = [designer.seed]
        best_ever = designer.seed
        
        def calculate_mape_percentage(molecule, target):
            """Calculate MAPE as percentage of target values."""
            target_dict = target.to_dict()
            props = molecule.properties
            
            errors = []
            for key in ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid']:
                if key in target_dict and key in props:
                    target_val = abs(target_dict[key])
                    if target_val > 0:
                        error_pct = abs(props[key] - target_dict[key]) / target_val * 100
                        errors.append(error_pct)
            
            return sum(errors) / len(errors) if errors else 100.0
        
        def get_mape_for_sorting(molecule):
            """Helper to get MAPE for sorting - lower is better."""
            return calculate_mape_percentage(molecule, target)
        
        print(f"[DEBUG] Starting beam search loop, max_iterations={config.beam_search.max_iterations}")
        print(f"[DEBUG] Initial beam size: {len(beam)}, best_ever: {best_ever.smiles if best_ever else 'None'}")
        
        for iteration in range(1, config.beam_search.max_iterations + 1):
            print(f"[DEBUG] === ITERATION {iteration} START ===")
            progress_queue.put({
                'type': 'status', 
                'message': f'Iteration {iteration}/{config.beam_search.max_iterations}'
            })
            
            # Generate candidates
            all_candidates = []
            for parent in beam:
                print(f"[DEBUG] Processing parent: {parent.smiles}")
                from agents.worker_agent import ChemistAgent
                agent = ChemistAgent(parent, target, config)
                candidates = agent.generate_variations()
                print(f"[DEBUG] Generated {len(candidates)} candidates from parent")
                all_candidates.extend(candidates)
            
            print(f"[DEBUG] Total candidates this iteration: {len(all_candidates)}")
            
            if not all_candidates:
                print(f"[DEBUG] No candidates generated! Exiting loop.")
                break
            
            # Filter and rank BY MAPE (lower is better)
            feasible = [c for c in all_candidates if c.is_feasible]
            unique = {c.smiles: c for c in feasible}.values()
            sorted_candidates = sorted(unique, key=get_mape_for_sorting)
            
            # Update beam - select top_k by MAPE
            beam = list(sorted_candidates)[:config.beam_search.top_k]
            
            # Update best_ever based on MAPE - check ALL candidates, not just beam
            if sorted_candidates:
                # sorted_candidates[0] has the best (lowest) MAPE
                best_this_iteration = list(sorted_candidates)[0]
                best_this_mape = get_mape_for_sorting(best_this_iteration)
                best_ever_mape = get_mape_for_sorting(best_ever)
                
                if best_this_mape < best_ever_mape:
                    best_ever = best_this_iteration
                    reporter.report_best(best_ever)
                
                # Check MAPE convergence
                mape_pct = calculate_mape_percentage(best_ever, target)
                print(f"[DEBUG] MAPE check: mape_pct={mape_pct}, threshold={mape_threshold_pct}, condition={mape_pct} <= {mape_threshold_pct}")
                progress_queue.put({
                    'type': 'status',
                    'message': f'MAPE: {mape_pct:.2f}% | Threshold: {mape_threshold_pct}%'
                })
                
                if mape_pct <= mape_threshold_pct:
                    print(f"[DEBUG] CONVERGENCE TRIGGERED: {mape_pct} <= {mape_threshold_pct}")
                    progress_queue.put({
                        'type': 'status',
                        'message': f'Converged! MAPE {mape_pct:.2f}% ≤ {mape_threshold_pct}%'
                    })
                    break
            
            # Report iteration with exactly beam_width candidates (not all)
            display_candidates = list(sorted_candidates)[:beam_width]
            reporter.report_iteration(iteration, display_candidates, beam, top_k, mape_calculator=get_mape_for_sorting)
        
        # Report final best molecule
        reporter.report_best(best_ever)
        reporter.report_complete()
        
    except Exception as e:
        progress_queue.put({'type': 'error', 'message': str(e)})
        import traceback
        print(traceback.format_exc())


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_search():
    """Start beam search optimization."""
    global current_search
    
    data = request.json
    
    target_props = {
        'density': float(data.get('density', 1.5)),
        'velocity': float(data.get('velocity', 7000)),
        'pressure': float(data.get('pressure', 20)),
        'hf': float(data.get('hf', 100))
    }
    
    enable_rag = data.get('enable_rag', True)
    beam_width = int(data.get('beam_width', 10))
    top_k = int(data.get('top_k', 5))
    max_iter = int(data.get('max_iter', 10))
    mape_threshold_pct = float(data.get('mape_threshold', 1.0))
    
    # Start search in background
    thread = threading.Thread(
        target=run_beam_search,
        args=(target_props, enable_rag, beam_width, top_k, max_iter, mape_threshold_pct),
        daemon=True
    )
    thread.start()
    current_search = thread
    
    return jsonify({'status': 'started'})


@app.route('/api/progress')
def progress_stream():
    """Stream progress updates using Server-Sent Events."""
    def generate():
        while True:
            if not progress_queue.empty():
                update = progress_queue.get()
                yield f"data: {json.dumps(update)}\n\n"
            else:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    # Run app on port 5001 (debug=False to avoid threading issues)
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║   ⚗️  ENERGETIC MOLECULAR DESIGN SYSTEM (EMDS)  ⚗️              ║")
    print("║                                                                  ║")
    print("║   🌐 Web GUI Server                                              ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("   🚀 Server Status: RUNNING")
    print()
    print("   🔗 Open in browser:")
    print("      http://localhost:5001")
    print()
    print("   ⌨️  Press Ctrl+C to stop the server")
    print()
    print("─" * 68)
    print()
    
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5001)
