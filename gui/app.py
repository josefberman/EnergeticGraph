"""
Flask web application for Molecular Design System GUI.
Provides a modern web interface for beam search optimization.
"""

import os
import sys
import json
import time
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from queue import Queue
import threading

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
    
    def report_iteration(self, iteration, candidates, beam, top_k):
        """Report iteration progress."""
        # Generate images for top candidates
        candidate_data = []
        for i, candidate in enumerate(candidates[:20]):  # Limit to top 20
            img_path = generate_molecule_image(
                candidate.smiles, 
                f'iter_{iteration}_cand_{i}.png',
                size=(150, 150)
            )
            candidate_data.append({
                'smiles': candidate.smiles,
                'properties': candidate.properties,
                'feasibility': candidate.feasibility,
                'score': candidate.score,
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


def run_beam_search(target_props, enable_rag, beam_width, top_k, max_iter):
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
        
        for iteration in range(1, config.beam_search.max_iterations + 1):
            progress_queue.put({
                'type': 'status', 
                'message': f'Iteration {iteration}/{config.beam_search.max_iterations}'
            })
            
            # Generate candidates
            all_candidates = []
            for parent in beam:
                from agents.worker_agent import ChemistAgent
                agent = ChemistAgent(parent, target, config)
                candidates = agent.generate_variations()
                all_candidates.extend(candidates)
            
            if not all_candidates:
                break
            
            # Filter and rank
            feasible = [c for c in all_candidates if c.is_feasible]
            unique = {c.smiles: c for c in feasible}.values()
            sorted_candidates = sorted(unique, key=lambda x: x.score)
            
            # Update beam
            beam = sorted_candidates[:config.beam_search.top_k]
            
            # Update best
            if beam and beam[0].score < best_ever.score:
                improvement = best_ever.score - beam[0].score
                best_ever = beam[0]
                reporter.report_best(best_ever)
                
                if improvement < config.beam_search.convergence_threshold:
                    progress_queue.put({'type': 'status', 'message': 'Converged!'})
                    break
            
            # Report iteration with top_k
            reporter.report_iteration(iteration, sorted_candidates, beam, top_k)
        
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
    
    # Start search in background
    thread = threading.Thread(
        target=run_beam_search,
        args=(target_props, enable_rag, beam_width, top_k, max_iter),
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
    print("=" * 60)
    print("  Molecular Design System - Web GUI")
    print("=" * 60)
    print(f"\n  Open in browser: http://localhost:5001\n")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5001)
