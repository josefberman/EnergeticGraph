from typing import Dict, Any, List
import os
import threading
import time
import base64
from io import BytesIO
from taipy.gui import Gui, State, notify
from rdkit import Chem
from rdkit.Chem import Draw
from molecular_optimizer_agent import MolecularOptimizationAgent


# Ensure Transformers does not try to import TensorFlow/Keras (avoids Keras 3 error)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


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
    except Exception:
        return ""


# Global state variables
uploaded_file = None
uploaded_path = ""
use_rag = False
error_metric = "mape"
beam_width = 5
max_iterations = 8
proceed_k = 3
start_mode = "From samples"
user_smiles = ""
is_running = False
cancel_event = None
worker_thread = None
result_holder = {}
results = None
show_results = False

# Display variables
start_mol_img = ""
best_mol_img = ""
start_smiles_text = ""
best_smiles_text = ""
target_props = {}
best_props = {}
best_score_text = ""
iterations_data = []
rag_info = ""


def background_run(csv_path: str, use_rag: bool, beam_width: int, max_iterations: int, 
                   proceed_k: int, result_holder: dict, cancel_event: threading.Event,
                   error_metric: str = 'mape', prefer_user_start: bool = False, 
                   starting_smiles: str = None):
    """Run optimization in background thread."""
    try:
        agent = MolecularOptimizationAgent(
            beam_width=beam_width,
            max_iterations=max_iterations,
            convergence_threshold=0.01,
            use_rag=use_rag,
            early_stop_patience=None,
            proceed_k=proceed_k,
            error_metric=error_metric,
            cli_rag_logging=bool(use_rag),
        )
        results = agent.process_csv_input(
            csv_path, 
            verbose=False, 
            cancel_event=cancel_event,
            starting_smiles=starting_smiles if starting_smiles else None,
            prefer_user_start=bool(prefer_user_start)
        )
        result_holder['results'] = results
        result_holder['completed'] = True
    except Exception as e:
        result_holder['results'] = {'error': str(e)}
        result_holder['completed'] = True


def on_file_upload(state: State, var_name: str, value):
    """Handle CSV file upload."""
    if value is not None:
        dest = os.path.join(os.getcwd(), 'uploaded_input.csv')
        try:
            with open(dest, 'wb') as f:
                f.write(value.read())
            state.uploaded_path = dest
            notify(state, 'success', f'✓ Uploaded: {os.path.basename(dest)}')
        except Exception as e:
            notify(state, 'error', f'Upload failed: {str(e)}')


def run_optimization(state: State):
    """Start optimization run."""
    global worker_thread, cancel_event, result_holder
    
    if not state.uploaded_path or not os.path.exists(state.uploaded_path):
        notify(state, 'warning', '⚠ Please upload a CSV file first')
        return
    
    if state.is_running:
        notify(state, 'info', 'Optimization already running')
        return
    
    # Reset results
    state.results = None
    state.show_results = False
    state.is_running = True
    
    # Setup worker
    cancel_event = threading.Event()
    result_holder = {'completed': False}
    
    prefer_user_start = state.start_mode == "Provide SMILES"
    smiles_input = state.user_smiles.strip() if prefer_user_start else None
    
    worker_thread = threading.Thread(
        target=background_run,
        args=(
            state.uploaded_path,
            state.use_rag,
            int(state.beam_width),
            int(state.max_iterations),
            int(state.proceed_k),
            result_holder,
            cancel_event,
            str(state.error_metric).lower(),
            prefer_user_start,
            smiles_input
        ),
        daemon=True
    )
    worker_thread.start()
    notify(state, 'info', '🚀 Optimization running in background...')
    
    # Start polling
    state.assign("poll_worker", True)


def poll_worker(state: State):
    """Poll worker thread status."""
    global worker_thread, result_holder
    
    if not state.is_running or not worker_thread:
        return
    
    if not worker_thread.is_alive():
        state.is_running = False
        if result_holder.get('completed'):
            state.results = result_holder.get('results')
            if state.results and 'error' not in state.results:
                state.show_results = True
                update_results_display(state)
                notify(state, 'success', '✓ Optimization complete!')
            elif state.results and 'error' in state.results:
                notify(state, 'error', f'Error: {state.results["error"]}')
        worker_thread = None
    else:
        # Continue polling
        time.sleep(1)
        state.assign("poll_worker", True)


def update_results_display(state: State):
    """Update the results display."""
    if not state.results:
        return
    
    results = state.results
    
    # Update molecule images
    state.start_smiles_text = results.get('starting_molecule', '')
    state.best_smiles_text = results.get('best_molecule', '')
    state.start_mol_img = smiles_to_base64(state.start_smiles_text)
    state.best_mol_img = smiles_to_base64(state.best_smiles_text)
    
    # Update properties
    state.target_props = results.get('target_properties', {})
    state.best_props = results.get('best_properties', {})
    
    # Update score
    best_score = results.get('best_score')
    if isinstance(best_score, (int, float)):
        state.best_score_text = f"Best Score: {best_score:.6f}"
    else:
        state.best_score_text = ""
    
    # Update RAG info if available
    rag_trace = results.get('rag_trace')
    if isinstance(rag_trace, dict):
        state.rag_info = f"RAG Query: {rag_trace.get('query', 'N/A')}"
    else:
        state.rag_info = ""


def reset_all(state: State):
    """Reset all state and clear uploaded files."""
    global worker_thread, cancel_event
    
    # Cancel running optimization
    if state.is_running and cancel_event and not cancel_event.is_set():
        cancel_event.set()
    
    # Delete uploaded file
    if state.uploaded_path and os.path.exists(state.uploaded_path):
        try:
            os.remove(state.uploaded_path)
        except Exception:
            pass
    
    # Reset all state
    state.uploaded_path = ""
    state.use_rag = False
    state.error_metric = "mape"
    state.beam_width = 5
    state.max_iterations = 8
    state.proceed_k = 3
    state.start_mode = "From samples"
    state.user_smiles = ""
    state.is_running = False
    state.results = None
    state.show_results = False
    state.start_mol_img = ""
    state.best_mol_img = ""
    state.best_score_text = ""
    state.target_props = {}
    state.best_props = {}
    
    notify(state, 'success', '✓ Reset complete')


# Define the page layout with modern design
page = """
<|container|
# ⚗️ EnergeticGraph Optimizer

Design and optimize energetic molecules with AI-powered beam search

<|layout|columns=300px 1fr|gap=2rem|

<|part|class_name=sidebar|
## Configuration

**Upload CSV File**

<|{uploaded_file}|file_selector|on_action=on_file_upload|extensions=.csv|label=Choose CSV|>

**Use RAG**

<|{use_rag}|toggle|>

**Error Metric**

<|{error_metric}|selector|lov=mape;mse|dropdown|>

**Beam Width**

<|{beam_width}|number|min=1|max=30|>

**Max Iterations**

<|{max_iterations}|number|min=1|max=100|>

**Proceeding Candidates (k)**

<|{proceed_k}|number|min=1|max=30|>

**Starting Molecule**

<|{start_mode}|selector|lov=From samples;Provide SMILES|dropdown|>

<|{start_mode == "Provide SMILES"}|part|
**Starting SMILES**

<|{user_smiles}|input|>
|>

<|Run Optimization|button|on_action=run_optimization|class_name=btn-primary|>

<|Reset All|button|on_action=reset_all|class_name=btn-secondary|>
|>

<|part|class_name=results-area|

<|{is_running}|part|
### 🔬 Optimization in Progress...

Running molecular beam search. Results will appear when complete.
|>

<|{not is_running and not show_results}|part|
### Ready to Optimize

Upload a CSV file and configure your optimization parameters to get started.
|>

<|{show_results and results}|part|
## ✨ Optimization Results

**{best_score_text}**

<|layout|columns=1fr 1fr|gap=2rem|

<|part|class_name=mol-card|
### Starting Molecule

<|{start_mol_img}|image|>

`{start_smiles_text}`
|>

<|part|class_name=mol-card|
### Best Molecule

<|{best_mol_img}|image|>

`{best_smiles_text}`
|>

|>

<|layout|columns=1fr 1fr|gap=2rem|

<|part|
### 🎯 Target Properties

<|{target_props}|table|>
|>

<|part|
### 📊 Best Molecule Properties

<|{best_props}|table|>
|>

|>

<|{rag_info != ""}|part|
**{rag_info}**
|>

|>

|>

|>

|>
"""

# CSS styling
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
}

.sidebar {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.results-area {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    min-height: 400px;
}

.mol-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.mol-card img {
    width: 100%;
    border-radius: 12px;
    background: white;
    padding: 1rem;
    margin-bottom: 1rem;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.9rem 1.5rem !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    width: 100% !important;
}

.btn-secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.9rem 1.5rem !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    margin-top: 0.8rem !important;
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3) !important;
    width: 100% !important;
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
}

input[type="number"],
select {
    width: 100%;
    padding: 0.8rem;
    border: 2px solid #e2e8f0;
    border-radius: 10px;
    transition: all 0.2s ease;
}

input[type="number"]:focus,
select:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}
"""

if __name__ == '__main__':
    gui = Gui(page=page, css_file=None)
    gui.run(
        title="EnergeticGraph Optimizer",
        port=5001,  # Changed to 5001 to avoid conflict
        dark_mode=False,
        use_reloader=False,
        stylekit={
            "color_primary": "#667eea",
            "color_secondary": "#764ba2"
        }
    )
