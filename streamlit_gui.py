from typing import Dict, Any, List
import os
# Ensure Transformers does not try to import TensorFlow/Keras (avoids Keras 3 error)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import threading
import time
import base64
from io import BytesIO
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from molecular_optimizer import MolecularOptimizationAgent



def smiles_to_png_bytes(smiles: str, width: int = 300, height: int = 220) -> bytes:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return b""
        img = Draw.MolToImage(mol, size=(width, height))
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception:
        return b""


def ensure_session_state():
    defaults = {
        'uploaded_path': '',
        'running': False,
        'cancel_event': None,
        'results': None,
        'worker': None,
        'result_holder': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def background_run(csv_path: str, use_rag: bool, beam_width: int, max_iterations: int, proceed_k: int, result_holder: dict, cancel_event: threading.Event | None, error_metric: str = 'mape', prefer_user_start: bool = False, starting_smiles: str | None = None):
    # Pure background execution: no Streamlit API here
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
        results = agent.process_csv_input(csv_path, verbose=False, cancel_event=cancel_event, starting_smiles=starting_smiles if starting_smiles else None, prefer_user_start=bool(prefer_user_start))
        result_holder['results'] = results
    except Exception as e:
        result_holder['results'] = {'error': str(e)}


def show_best(results: Dict[str, Any], metric: str = 'mape'):
    start_smiles = results.get('starting_molecule', '')
    best_smiles = results.get('best_molecule', '')
    target_props = results.get('target_properties', {})
    best_props = results.get('best_properties', {})

    st.subheader('Best Result')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('Starting Molecule')
        png = smiles_to_png_bytes(start_smiles)
        if png:
            st.image(png)
        st.code(start_smiles, language=None)
    with col2:
        st.caption('Best Molecule')
        png = smiles_to_png_bytes(best_smiles)
        if png:
            st.image(png)
        st.code(best_smiles, language=None)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Target Properties**')
        if target_props:
            st.table({k: [v] for k, v in target_props.items()})
        else:
            st.info('No target properties')
    with c2:
        st.markdown('**Best Molecule Properties**')
        if best_props:
            st.table({k: [v] for k, v in best_props.items()})
        else:
            st.info('No properties')

    # Show best score summary and components
    best_score = results.get('best_score', None)
    best_prop_error = results.get('best_prop_error', None)
    best_feas = results.get('best_feasibility_score', None)
    cols = st.columns(2)
    with cols[0]:
        if isinstance(best_score, (int, float)):
            st.markdown(f"**Best score ({str(metric).upper()}):** `{best_score:.6f}`")
    with cols[1]:
        lines = []
        if isinstance(best_prop_error, (int, float)):
            lines.append(f"prop_err: `{best_prop_error:.6f}`")
        if isinstance(best_feas, (int, float)):
            lines.append(f"feas: `{best_feas:.3f}`")
        if lines:
            st.markdown("**Components:** " + " | ".join(lines))


def show_history(results: Dict[str, Any], metric: str = 'mape'):
    st.subheader('Beam Search (per iteration; error is MAPE/MSE, feasibility filtered)')
    history: List[Dict[str, Any]] = results.get('search_history', []) or []
    if not history:
        st.info('No search history available')
        return
    for entry in history:
        it = entry.get('iteration')
        candidates = entry.get('candidates', [])
        with st.expander(f'Iteration {it} (top {len(candidates)})', expanded=False):
            cols = st.columns(3)
            for idx, c in enumerate(candidates):
                smiles = c.get('smiles', '')
                score = c.get('score')
                png = smiles_to_png_bytes(smiles)
                with cols[idx % 3]:
                    if png:
                        st.image(png, width='stretch')
                    st.caption(f'SMILES: {smiles[:80]}{"..." if len(smiles) > 80 else ""}')
                    feas_score = c.get('feasibility_score', None)
                    if isinstance(feas_score, (int, float)):
                        st.markdown(f"*Feasibility (0-1):* `{float(feas_score):.3f}`")
                    else:
                        feas = c.get('feasibility') or {}
                        if isinstance(feas, dict) and 'composite_score_0_1' in feas:
                            st.markdown(f"*Feasibility (0-1):* `{float(feas['composite_score_0_1']):.3f}`")
                        else:
                            st.markdown("*Feasibility (0-1):* `N/A`")
                    pe = c.get('prop_error')
                    if isinstance(pe, (int, float)):
                        st.markdown(f"*Error ({str(metric).upper()}):* `{pe:.6f}`")
                    props = c.get('properties') or {}
                    if isinstance(props, dict) and props:
                        try:
                            st.markdown('*Predicted properties:*')
                            st.table({k: [v] for k, v in props.items()})
                        except Exception:
                            pass


def main():
    st.set_page_config(page_title='EnergeticGraph Optimizer', layout='wide')
    ensure_session_state()

    st.title('EnergeticGraph Optimizer')
    st.caption('Design and optimize energetic molecules. Upload a CSV, configure search, and run the optimizer.')

    with st.sidebar:
        st.header('Input & Settings')
        uploaded = st.file_uploader('Upload CSV', type=['csv'])
        if uploaded is not None:
            dest = os.path.join(os.getcwd(), 'uploaded_input.csv')
            with open(dest, 'wb') as f:
                f.write(uploaded.read())
            st.session_state.uploaded_path = dest
            st.success(f'Uploaded: {os.path.basename(dest)}')

        use_rag = st.toggle('Use RAG', value=False)
        metric = st.selectbox('Error metric', options=['mape', 'mse'], index=0)
        beam_width = st.number_input('Beam width', value=5, min_value=1, max_value=30, step=1)
        max_iterations = st.number_input('Max iterations', value=8, min_value=1, max_value=100, step=1)
        proceed_k = st.number_input('Proceeding candidates (k)', value=3, min_value=1, max_value=30, step=1)

        st.markdown('---')
        start_mode = st.radio('Starting molecule', options=['From samples', 'Provide SMILES'], index=0, horizontal=False)
        user_smiles = ''
        prefer_user_start = False
        if start_mode == 'Provide SMILES':
            user_smiles = st.text_input('Starting SMILES', value='', placeholder='e.g., CC1=CC=C(C=C1)[N+](=O)[O-]')
            prefer_user_start = True

        run_btn = st.button('Run Optimization', type='primary', width='stretch', disabled=st.session_state.running)
        stop_btn = st.button('Stop', width='stretch')
        reset_btn = st.button('Reset', width='stretch')

    # Poll worker status first
    if st.session_state.running and st.session_state.worker:
        if not st.session_state.worker.is_alive():
            # collect results
            st.session_state.running = False
            if st.session_state.result_holder:
                st.session_state.results = st.session_state.result_holder.get('results')
            st.session_state.worker = None
            st.session_state.cancel_event = None
            st.session_state.result_holder = None

    if run_btn and not st.session_state.running:
        csv_path = st.session_state.uploaded_path
        if not csv_path or not os.path.exists(csv_path):
            st.warning('Please upload a CSV file first')
        else:
            st.session_state.cancel_event = threading.Event()
            st.session_state.result_holder = {}
            worker = threading.Thread(
                target=background_run,
                args=(csv_path, use_rag, int(beam_width), int(max_iterations), int(proceed_k), st.session_state.result_holder, st.session_state.cancel_event, str(metric).lower(), bool(prefer_user_start), str(user_smiles).strip() if prefer_user_start and user_smiles else None),
                daemon=True,
            )
            st.session_state.worker = worker
            st.session_state.running = True
            worker.start()
            st.rerun()

    if stop_btn and st.session_state.running:
        try:
            if st.session_state.cancel_event and not st.session_state.cancel_event.is_set():
                st.session_state.cancel_event.set()
                st.info('Cancellation requested...')
        except Exception:
            pass

    if reset_btn:
        try:
            if st.session_state.running and st.session_state.cancel_event and not st.session_state.cancel_event.is_set():
                st.session_state.cancel_event.set()
        except Exception:
            pass
        # delete uploaded file
        path = st.session_state.uploaded_path
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        st.session_state.uploaded_path = ''
        st.session_state.results = None
        st.session_state.running = False
        st.session_state.cancel_event = None
        st.session_state.worker = None
        st.session_state.result_holder = None
        st.success('Reset complete')
        st.rerun()

    # Main content & auto-poll while running
    if st.session_state.running:
        st.info('Running in background; results will appear when finished.')
        time.sleep(1.0)
        try:
            st.rerun()
        except Exception:
            try:
                # Fallback for very old Streamlit versions
                getattr(st, 'experimental_rerun')()
            except Exception:
                pass

    results = st.session_state.results
    if results and 'error' in results:
        st.error(f"Error: {results['error']}")
    elif results and not st.session_state.running:
        show_best(results, metric=str(metric).lower() if 'metric' in locals() else 'mape')
        st.divider()
        show_history(results, metric=str(metric).lower() if 'metric' in locals() else 'mape')
        # Show RAG trace if available
        rag_trace = results.get('rag_trace') if isinstance(results, dict) else None
        if isinstance(rag_trace, dict):
            st.subheader('RAG Retrieval Trace')
            st.markdown(f"**Query:** {rag_trace.get('query','')}")
            titles = rag_trace.get('retrieved_titles') or []
            st.markdown(f"**Retrieved articles ({rag_trace.get('retrieved_count',0)}):**")
            if titles:
                for t in titles:
                    st.markdown(f"- {t}")
            st.markdown(f"**Names extracted:** `{rag_trace.get('names_extracted',0)}` | **Names converted:** `{rag_trace.get('names_converted',0)}`")
            st.markdown(f"**SMILES extracted:** `{rag_trace.get('smiles_extracted',0)}` | **Candidates scored:** `{rag_trace.get('candidates_scored',0)}`")
            if rag_trace.get('fallback_used'):
                st.info('RAG yielded no molecules; fell back to local CSV candidates')
            preview = rag_trace.get('top_preview') or []
            if preview:
                st.markdown('**Top candidates (preview):**')
                st.table({ 'smiles': [p.get('smiles') for p in preview], 'error': [p.get('score') for p in preview] })


if __name__ == '__main__':
    main()


