from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import threading
import time
import base64
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from molecular_optimizer_agent import MolecularOptimizationAgent

# Ensure Transformers does not try to import TensorFlow/Keras
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'energetic_graph_optimizer_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for optimization
optimization_state = {
    'running': False,
    'results': None,
    'cancel_event': None,
    'worker': None,
    'uploaded_file': None
}


def smiles_to_base64(smiles: str, width: int = 300, height: int = 220) -> str:
    """Convert SMILES to base64-encoded PNG image."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        img = Draw.MolToImage(mol, size=(width, height))
        buf = BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Error rendering SMILES: {e}")
        return ""


def background_optimization(csv_path, config, result_holder, cancel_event):
    """Run optimization in background thread with WebSocket updates."""
    try:
        socketio.emit('status_update', {'status': 'Starting optimization...'})
        
        agent = MolecularOptimizationAgent(
            beam_width=config['beam_width'],
            max_iterations=config['max_iterations'],
            convergence_threshold=0.01,
            use_rag=config['use_rag'],
            early_stop_patience=None,
            proceed_k=config['proceed_k'],
            error_metric=config['error_metric'],
            cli_rag_logging=bool(config['use_rag']),
        )
        
        socketio.emit('status_update', {'status': 'Running beam search optimization...'})
        
        results = agent.process_csv_input(
            csv_path,
            verbose=False,
            cancel_event=cancel_event,
            starting_smiles=config.get('starting_smiles'),
            prefer_user_start=bool(config.get('starting_smiles'))
        )
        
        result_holder['results'] = results
        result_holder['completed'] = True
        
        socketio.emit('optimization_complete', {'success': True})
        
    except Exception as e:
        result_holder['results'] = {'error': str(e)}
        result_holder['completed'] = True
        socketio.emit('optimization_complete', {'success': False, 'error': str(e)})


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'File must be a CSV'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.csv')
        file.save(filepath)
        
        optimization_state['uploaded_file'] = filepath
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/optimize', methods=['POST'])
def start_optimization():
    """Start the optimization process."""
    global optimization_state
    
    if optimization_state['running']:
        return jsonify({'success': False, 'error': 'Optimization already running'}), 400
    
    if not optimization_state['uploaded_file'] or not os.path.exists(optimization_state['uploaded_file']):
        return jsonify({'success': False, 'error': 'Please upload a CSV file first'}), 400
    
    try:
        data = request.json
        config = {
            'use_rag': data.get('use_rag', False),
            'error_metric': data.get('error_metric', 'mape').lower(),
            'beam_width': int(data.get('beam_width', 5)),
            'max_iterations': int(data.get('max_iterations', 8)),
            'proceed_k': int(data.get('proceed_k', 3)),
            'starting_smiles': data.get('starting_smiles', '').strip() or None
        }
        
        # Setup optimization
        cancel_event = threading.Event()
        result_holder = {'completed': False}
        
        worker = threading.Thread(
            target=background_optimization,
            args=(optimization_state['uploaded_file'], config, result_holder, cancel_event),
            daemon=True
        )
        
        optimization_state.update({
            'running': True,
            'results': None,
            'cancel_event': cancel_event,
            'worker': worker,
            'result_holder': result_holder
        })
        
        worker.start()
        
        return jsonify({'success': True, 'message': 'Optimization started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current optimization status."""
    global optimization_state
    
    is_running = optimization_state['running']
    
    # Check if worker finished
    if is_running and optimization_state['worker'] and not optimization_state['worker'].is_alive():
        optimization_state['running'] = False
        is_running = False
    
    return jsonify({
        'running': is_running,
        'has_results': optimization_state.get('result_holder', {}).get('completed', False)
    })


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get optimization results."""
    global optimization_state
    
    if not optimization_state.get('result_holder') or not optimization_state['result_holder'].get('completed'):
        return jsonify({'success': False, 'error': 'No results available'}), 404
    
    results = optimization_state['result_holder'].get('results', {})
    
    if 'error' in results:
        return jsonify({'success': False, 'error': results['error']}), 500
    
    # Prepare results with molecule images
    start_smiles = results.get('starting_molecule', '')
    best_smiles = results.get('best_molecule', '')
    
    response_data = {
        'success': True,
        'start_molecule': {
            'smiles': start_smiles,
            'image': smiles_to_base64(start_smiles)
        },
        'best_molecule': {
            'smiles': best_smiles,
            'image': smiles_to_base64(best_smiles)
        },
        'target_properties': results.get('target_properties', {}),
        'best_properties': results.get('best_properties', {}),
        'best_score': results.get('best_score'),
        'best_prop_error': results.get('best_prop_error'),
        'best_feasibility_score': results.get('best_feasibility_score'),
        'search_history': [],
        'rag_trace': results.get('rag_trace')
    }
    
    # Process search history
    history = results.get('search_history', []) or []
    for entry in history:
        iteration_data = {
            'iteration': entry.get('iteration'),
            'candidates': []
        }
        
        for candidate in entry.get('candidates', []):
            smiles = candidate.get('smiles', '')
            iteration_data['candidates'].append({
                'smiles': smiles,
                'image': smiles_to_base64(smiles, width=250, height=200),
                'score': candidate.get('score'),
                'feasibility_score': candidate.get('feasibility_score'),
                'prop_error': candidate.get('prop_error'),
                'properties': candidate.get('properties', {})
            })
        
        response_data['search_history'].append(iteration_data)
    
    return jsonify(response_data)


@app.route('/api/cancel', methods=['POST'])
def cancel_optimization():
    """Cancel running optimization."""
    global optimization_state
    
    if optimization_state['running'] and optimization_state['cancel_event']:
        optimization_state['cancel_event'].set()
        return jsonify({'success': True, 'message': 'Cancellation requested'})
    
    return jsonify({'success': False, 'error': 'No optimization running'}), 400


@app.route('/api/reset', methods=['POST'])
def reset_state():
    """Reset all state."""
    global optimization_state
    
    # Cancel if running
    if optimization_state['running'] and optimization_state['cancel_event']:
        optimization_state['cancel_event'].set()
    
    # Delete uploaded file
    if optimization_state['uploaded_file'] and os.path.exists(optimization_state['uploaded_file']):
        try:
            os.remove(optimization_state['uploaded_file'])
        except Exception:
            pass
    
    # Reset state
    optimization_state.update({
        'running': False,
        'results': None,
        'cancel_event': None,
        'worker': None,
        'uploaded_file': None,
        'result_holder': None
    })
    
    return jsonify({'success': True, 'message': 'Reset complete'})


if __name__ == '__main__':
    print("🚀 Starting EnergeticGraph Optimizer Web GUI...")
    print("📡 Server will be available at: http://localhost:5002")
    print("Press Ctrl+C to stop the server")
    
    socketio.run(app, host='0.0.0.0', port=5002, debug=False, allow_unsafe_werkzeug=True)
