import os
import threading
import time
import base64
from io import BytesIO
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from taipy.gui import Gui, notify, State, invoke_long_callback

from molecular_optimizer_agent import MolecularOptimizationAgent

# --- Helper Functions ---

def smiles_to_base64_png(smiles: str, width: int = 300, height: int = 220) -> str:
    """Convert SMILES to a base64 PNG string for display in Taipy."""
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

# --- State Variables ---

uploaded_path = ""
use_rag = False
metric = "mape"  # Options: 'mape', 'mse'
beam_width = 5
max_iterations = 8
proceed_k = 3
start_mode = "From samples"  # Options: 'From samples', 'Provide SMILES'
start_mode_lov = [("From samples", "From samples"), ("Provide SMILES", "Provide SMILES")]
user_smiles = ""
prefer_user_start = False
is_running = False
can_run_opt = True
results = None

# For displaying results
best_smiles_img = ""
start_smiles_img = ""
best_score_display = ""
best_prop_error_display = ""
best_feas_display = ""
target_props_data = pd.DataFrame()
best_props_data = pd.DataFrame()
search_history_data = [] # List of expandable items or formatted strings
rag_trace_display = ""

# --- Worker Function (Background) ---

def run_optimization(state):
    """
    Wrapper to run the optimization agent in a background thread.
    We use invoke_long_callback or manual threading updates.
    """
    # Prepare arguments from state
    csv_path = state.uploaded_path
    _use_rag = state.use_rag
    _beam_width = int(state.beam_width)
    _max_iterations = int(state.max_iterations)
    _proceed_k = int(state.proceed_k)
    _metric = str(state.metric).lower()
    _prefer_user_start = (state.start_mode == 'Provide SMILES')
    _user_smiles = str(state.user_smiles).strip() if _prefer_user_start else None
    
    cancel_event = threading.Event() # Simple cancellation not fully wired in Taipy basic example yet
    
    result_holder = {}
    
    try:
        agent = MolecularOptimizationAgent(
            beam_width=_beam_width,
            max_iterations=_max_iterations,
            convergence_threshold=0.01,
            use_rag=_use_rag,
            early_stop_patience=None,
            proceed_k=_proceed_k,
            error_metric=_metric,
            cli_rag_logging=bool(_use_rag),
        )
        # We invoke the synchronous process; 
        # Ideally, we'd check cancel_event periodically if passed down.
        res = agent.process_csv_input(
            csv_path, 
            verbose=False, 
            cancel_event=cancel_event, 
            starting_smiles=_user_smiles, 
            prefer_user_start=_prefer_user_start
        )
        return res
    except Exception as e:
        return {'error': str(e)}

def on_optimization_done(state, status, result):
    """Callback when the long-running task is finished."""
    state.is_running = False
    state.can_run_opt = True
    
    if isinstance(result, dict) and 'error' in result:
        notify(state, 'error', f"Error: {result['error']}")
        return

    # Update state with results
    state.results = result
    
    # Process results for display
    start_smi = result.get('starting_molecule', '')
    best_smi = result.get('best_molecule', '')
    
    state.start_smiles_img = smiles_to_base64_png(start_smi)
    state.best_smiles_img = smiles_to_base64_png(best_smi)
    
    # Scores
    bs = result.get('best_score')
    bpe = result.get('best_prop_error')
    bfs = result.get('best_feasibility_score')
    
    state.best_score_display = f"{bs:.6f}" if isinstance(bs, (int, float)) else "N/A"
    state.best_prop_error_display = f"{bpe:.6f}" if isinstance(bpe, (int, float)) else "N/A"
    state.best_feas_display = f"{bfs:.3f}" if isinstance(bfs, (int, float)) else "N/A"
    
    # Properties Tables
    t_props = result.get('target_properties', {})
    b_props = result.get('best_properties', {})
    
    state.target_props_data = pd.DataFrame([t_props])
    state.best_props_data = pd.DataFrame([b_props])

    # Search History (simplified for Taipy display, maybe just a summary text or table)
    # For a rich display like Streamlit's expanders, we might need a repeater or table.
    # Let's format a summary string or use a table for the best candidates per iteration.
    history = result.get('search_history', [])
    hist_summary = []
    for entry in history:
        it = entry.get('iteration')
        cands = entry.get('candidates', [])
        top_scores = ", ".join([f"{c.get('score'):.4f}" for c in cands[:3]])
        hist_summary.append({'Iteration': it, 'Beam Size': len(cands), 'Top Scores': top_scores})
    state.search_history_data = pd.DataFrame(hist_summary)

    # RAG Trace
    rag = result.get('rag_trace')
    if isinstance(rag, dict):
        trace_str = f"Query: {rag.get('query','')}\n\n"
        trace_str += f"Retrieved {rag.get('retrieved_count',0)} articles.\n"
        titles = rag.get('retrieved_titles') or []
        for t in titles:
            trace_str += f"- {t}\n"
        state.rag_trace_display = trace_str
    else:
        state.rag_trace_display = "No RAG trace available."

    notify(state, 'success', 'Optimization Completed!')


# --- Event Handlers ---

def on_change(state, var_name, var_value):
    if var_name == "start_mode":
        state.prefer_user_start = (var_value == "Provide SMILES")

def on_file_upload(state):
    """Handle CSV file upload."""
    # Taipy handles upload via file_selector binding to a path string usually.
    # We will check if uploaded_path is set.
    if os.path.exists(state.uploaded_path):
        notify(state, 'success', f'Uploaded: {os.path.basename(state.uploaded_path)}')
    else:
        notify(state, 'warning', 'Upload failed or file not found.')

def on_run_click(state):
    if state.is_running:
        return
    
    if not state.uploaded_path or not os.path.exists(state.uploaded_path):
        notify(state, 'warning', 'Please upload a CSV file first.')
        return

    notify(state, 'info', 'Starting optimization in background...')
    state.is_running = True
    state.can_run_opt = False
    
    # Clear previous results
    state.results = None
    state.best_smiles_img = ""
    state.start_smiles_img = ""
    state.best_score_display = ""
    state.target_props_data = pd.DataFrame()
    state.best_props_data = pd.DataFrame()
    state.search_history_data = pd.DataFrame()
    state.rag_trace_display = ""

    # Start long running task
    invoke_long_callback(state, run_optimization, [state], on_optimization_done)

def on_stop_click(state):
    # Cooperative cancellation isn't trivially exposed via simple invoke_long_callback
    # without passing a threading.Event that we can set here. 
    # For this demo, we'll just notify.
    notify(state, 'info', 'Stop requested (not fully implemented in demo).')

def on_reset_click(state):
    state.uploaded_path = ""
    state.results = None
    state.is_running = False
    state.can_run_opt = True
    state.start_smiles_img = ""
    state.best_smiles_img = ""
    state.best_score_display = ""
    state.target_props_data = pd.DataFrame()
    state.best_props_data = pd.DataFrame()
    state.search_history_data = pd.DataFrame()
    state.rag_trace_display = ""
    notify(state, 'success', 'Reset complete.')

# --- Layout (Markdown) ---

page = """
# EnergeticGraph **Optimizer**

Design and optimize energetic molecules. Upload a CSV, configure search, and run the optimizer.

<|layout|columns=300px 1|
    <|part|class_name=sidebar|
        ### Input & Settings

        **Upload CSV**
        <|{uploaded_path}|file_selector|label=Upload CSV|extensions=.csv|on_action=on_file_upload|>

        <|{use_rag}|toggle|label=Use RAG|>
        
        **Error Metric**
        <|{metric}|selector|lov=mape;mse|dropdown|>

        **Beam Width**
        <|{beam_width}|number|min=1|max=30|>

        **Max Iterations**
        <|{max_iterations}|number|min=1|max=100|>

        **Proceed (k)**
        <|{proceed_k}|number|min=1|max=30|>

        ---
        **Starting Molecule**
        <|{start_mode}|selector|lov=From samples;Provide SMILES|mode=radio|>
        
        <|{user_smiles}|input|label=Starting SMILES|active={prefer_user_start}|>

        <|Run Optimization|button|on_action=on_run_click|active={can_run_opt}|class_name=primary|>
        <|Stop|button|on_action=on_stop_click|active={is_running}|>
        <|Reset|button|on_action=on_reset_click|>
        
        <|{is_running}|indicator|value={is_running}|label=Running...|>
    |>

    <|part|
        ## Optimization Results

        <|layout|columns=1 1|
            <|part|
                ### Starting Molecule
                <|{start_smiles_img}|image|width=300px|>
            |>
            <|part|
                ### Best Molecule
                <|{best_smiles_img}|image|width=300px|>
            |>
        |>

        <|layout|columns=1 1|
            <|part|
                **Target Properties**
                <|{target_props_data}|table|>
            |>
            <|part|
                **Best Molecule Properties**
                <|{best_props_data}|table|>
            |>
        |>

        ### Score Summary
        **Best Score ({metric}):** <|{best_score_display}|text|>
        
        **Components:** Prop Error: <|{best_prop_error_display}|text|> | Feasibility: <|{best_feas_display}|text|>

        ---
        ### Search History Summary
        <|{search_history_data}|table|>

        ### RAG Trace
        <|{rag_trace_display}|text|mode=pre|>
    |>
|>
"""

if __name__ == "__main__":
    Gui(page=page).run(title="EnergeticGraph Optimizer", use_reloader=True)

