from typing import List, Dict, Any
import asyncio
import os
import base64
from io import BytesIO
from nicegui import ui
from nicegui.events import UploadEventArguments
from rdkit import Chem
from rdkit.Chem import Draw
from molecular_optimizer_agent import MolecularOptimizationAgent


def smiles_to_data_uri(smiles: str, width: int = 250, height: int = 200) -> str:
    """Render SMILES to a PNG data URI using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        img = Draw.MolToImage(mol, size=(width, height))
        buf = BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{b64}'
    except Exception:
        return ''


def build_iteration_view(search_history: List[Dict[str, Any]]):
    """Create a visual view per iteration with candidate molecules and scores."""
    for entry in search_history:
        it = entry.get('iteration')
        candidates = entry.get('candidates', [])
        with ui.expansion(f'Iteration {it} (top {len(candidates)})', value=False):
            with ui.grid(columns=3).classes('gap-6'):
                for c in candidates:
                    smiles = c.get('smiles', '')
                    score = c.get('score', None)
                    img_uri = smiles_to_data_uri(smiles)
                    with ui.card().tight():
                        ui.image(img_uri).classes('w-64 h-52')
                        ui.label(f'SMILES: {smiles[:40]}{"..." if len(smiles) > 40 else ""}').classes('text-xs break-all')
                        if score is not None:
                            ui.label(f'Score (MSE): {score:.4f}').classes('text-sm')


def build_properties_table(title: str, props: Dict[str, float]):
    """Show a simple property key-value table."""
    with ui.card().classes('w-full'):
        ui.label(title).classes('text-lg font-bold')
        if not props:
            ui.label('No properties').classes('text-sm')
            return
        columns = [
            {'name': 'name', 'label': 'Property', 'field': 'name'},
            {'name': 'value', 'label': 'Value', 'field': 'value'},
        ]
        rows = [{'name': k, 'value': v} for k, v in props.items()]
        ui.table(columns=columns, rows=rows).props('flat dense')


async def run_optimization(csv_path: str, use_rag: bool, beam_width: int, max_iterations: int, proceed_k: int, cancel_event: asyncio.Event) -> Dict[str, Any]:
    """Run optimization in a thread to avoid blocking the UI."""
    loop = asyncio.get_running_loop()
    def _blocking_call():
        # Disable early stopping so GUI respects max_iterations fully
        agent = MolecularOptimizationAgent(beam_width=beam_width, max_iterations=max_iterations, convergence_threshold=0.01, use_rag=use_rag, early_stop_patience=None, proceed_k=proceed_k)
        return agent.process_csv_input(csv_path, verbose=False, cancel_event=cancel_event)
    return await loop.run_in_executor(None, _blocking_call)


def run_gui() -> None:
    """Start the NiceGUI application."""
    ui.page_title('EnergeticGraph Optimizer')
    ui.markdown('**EnergeticGraph Optimizer**: Upload a CSV, run beam search, and visualize candidates and the best molecule.').classes('mb-4')

    uploaded_path_holder = {'path': ''}
    current_cancel_event: asyncio.Event | None = None

    with ui.row().classes('items-start w-full gap-8'):
        with ui.card().classes('min-w-[380px]'):
            ui.label('Input').classes('text-lg font-bold')

            # File upload (wrapped to allow reset by re-creating component)
            upload_container = ui.column()
            def make_uploader():
                def on_upload(e: UploadEventArguments):
                    dest = os.path.join(os.getcwd(), 'uploaded_input.csv')
                    with open(dest, 'wb') as f:
                        f.write(e.content.read())
                    uploaded_path_holder['path'] = dest
                    ui.notify(f'Uploaded: {os.path.basename(dest)}')
                return ui.upload(on_upload=on_upload, auto_upload=True, label='Upload CSV').props('accept=.csv')
            with upload_container:
                upload_ref = make_uploader()

            # Options
            rag_switch = ui.switch('Use RAG', value=False)
            beam_width_input = ui.number('Beam width', value=5, min=1, max=30, step=1).classes('w-40')
            max_iter_input = ui.number('Max iterations', value=8, min=1, max=50, step=1).classes('w-40')
            proceed_k_input = ui.number('Proceeding candidates (k)', value=3, min=1, max=20, step=1).classes('w-40')
            with ui.row().classes('gap-2'):
                run_button = ui.button('Run Optimization', icon='play_arrow', color='primary')
                reset_button = ui.button('Reset', icon='refresh', color='warning')

        with ui.column().classes('w-full gap-6') as results_column:
            # placeholders
            best_section = ui.expansion('Best Result', value=False)
            with best_section:
                ui.label('Run to see results')

            ui.separator()
            ui.label('Beam Search (per iteration)').classes('text-lg font-bold')
            beam_container = ui.column().classes('w-full')

    async def handle_run_click():
        csv_path = uploaded_path_holder['path']
        if not csv_path or not os.path.exists(csv_path):
            ui.notify('Please upload a CSV file first', color='warning')
            return

        run_button.disable()
        ui.notify('Running optimization...')
        try:
            # Set up cancel event for this run
            nonlocal current_cancel_event
            current_cancel_event = asyncio.Event()
            results = await run_optimization(csv_path, bool(rag_switch.value), int(beam_width_input.value), int(max_iter_input.value), int(proceed_k_input.value), current_cancel_event)
        except Exception as e:
            run_button.enable()
            ui.notify(f'Error: {e}', color='negative')
            return

        run_button.enable()
        if 'error' in results:
            ui.notify(results['error'], color='negative')
            return

        # Clear previous content
        best_section.clear()
        beam_container.clear()

        # Best results
        with best_section:
            start_smiles = results.get('starting_molecule', '')
            best_smiles = results.get('best_molecule', '')
            target_props = results.get('target_properties', {})
            best_props = results.get('best_properties', {})
            with ui.row().classes('gap-10'):
                with ui.card():
                    ui.label('Starting Molecule')
                    ui.image(smiles_to_data_uri(start_smiles)).classes('w-64 h-52')
                    ui.label(start_smiles).classes('text-xs break-all')
                with ui.card():
                    ui.label('Best Molecule')
                    ui.image(smiles_to_data_uri(best_smiles)).classes('w-64 h-52')
                    ui.label(best_smiles).classes('text-xs break-all')
            with ui.row().classes('gap-8 w-full'):
                build_properties_table('Target Properties', target_props)
                build_properties_table('Best Molecule Properties', best_props)

        best_section.value = True

        # Beam search history
        search_history = results.get('search_history', [])
        if not search_history:
            ui.notify('No search history available', color='warning')
        else:
            build_iteration_view(search_history)

    def handle_reset_click():
        # Cancel any ongoing run
        try:
            if current_cancel_event is not None and not current_cancel_event.is_set():
                current_cancel_event.set()
        except Exception:
            pass
        # Delete uploaded file if exists
        path = uploaded_path_holder.get('path')
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        uploaded_path_holder['path'] = ''
        # Reset inputs
        try:
            rag_switch.value = False
            beam_width_input.value = 5
            max_iter_input.value = 8
            proceed_k_input.value = 3
        except Exception:
            pass
        # Clear result containers and reset uploader
        best_section.clear()
        with best_section:
            ui.label('Run to see results')
        best_section.value = False
        beam_container.clear()
        # Reset upload component by recreating it
        try:
            upload_container.clear()
            with upload_container:
                upload_ref = make_uploader()
        except Exception:
            pass
        ui.notify('Reset complete')

    run_button.on('click', handle_run_click)
    reset_button.on('click', handle_reset_click)

    ui.run(title='EnergeticGraph Optimizer', reload=False)
