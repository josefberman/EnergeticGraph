# Suppress RDKit warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

from typing import List, Dict, Any, Optional
import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from prediction import predict_properties, convert_name_to_smiles
from molecular_tools import (
    validate_molecule_structure, 
    generate_molecular_modifications, 
    calculate_molecular_descriptors,
    check_energetic_functional_groups
)
from RAG import retrieve_context
import os
from dotenv import load_dotenv
import math
import json
from dataclasses import dataclass

load_dotenv()

# Suppress RDKit error logs during parsing to reduce console noise
try:
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass

@dataclass
class OptimizationState:
    """State for the molecular optimization process"""
    current_molecule: str
    target_properties: Dict[str, float]
    weights: Dict[str, float]
    beam_candidates: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    beam_width: int
    best_score: float
    best_molecule: str
    best_gibbs: Optional[float]
    search_history: List[Dict[str, Any]]
    convergence_threshold: float
    verbose: bool

class MolecularOptimizationAgent:
    """Enhanced molecular optimization agent using LangGraph"""
    
    def __init__(self, beam_width: int = 5, max_iterations: int = 10, convergence_threshold: float = 0.01, use_rag: bool = True, early_stop_patience: int | None = 3, proceed_k: int = 3, error_metric: str = 'mape', feasibility_weight: float = 0.3, feasibility_threshold: float = 0.4):
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_rag = use_rag
        self.early_stop_patience = early_stop_patience
        self.proceed_k = proceed_k
        self.error_metric = (error_metric or 'mape').lower()
        # Weight for feasibility in combined score: score = (1-w)*prop_error + w*(1-feas)
        try:
            w = float(feasibility_weight)
        except Exception:
            w = 0.3
        self.feasibility_weight = max(0.0, min(1.0, w))
        # Minimum feasibility required for candidates (composite_score_0_1)
        try:
            self.feasibility_threshold = float(feasibility_threshold)
        except Exception:
            self.feasibility_threshold = 0.4
        
        # Initialize the LLM with all available tools
        self.model = ChatOpenAI(model='gpt-4o', temperature=0).bind_tools([
            predict_properties, 
            convert_name_to_smiles,
            validate_molecule_structure,
            generate_molecular_modifications,
            calculate_molecular_descriptors,
            check_energetic_functional_groups,
            retrieve_context
        ])
        
        # Initialize the workflow
        self.workflow = self._create_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for molecular optimization"""
        
        def call_model(state: MessagesState):
            return {'messages': self.model.invoke(state['messages'])}
        
        def should_continue(state: MessagesState):
            last_message = state['messages'][-1]
            if last_message.tool_calls:
                return 'tools'
            return END
        
        workflow = StateGraph(MessagesState)
        workflow.add_node('agent', call_model)
        workflow.add_node('tools', ToolNode([
            predict_properties, 
            convert_name_to_smiles,
            validate_molecule_structure,
            generate_molecular_modifications,
            calculate_molecular_descriptors,
            check_energetic_functional_groups,
            retrieve_context
        ]))
        workflow.add_edge(START, 'agent')
        workflow.add_conditional_edges('agent', should_continue)
        workflow.add_edge('tools', 'agent')
        
        return workflow

    # ---------- Feasibility scoring (replaces Gibbs usage)
    @dataclass
    class FeasibilityReport:
        sa_score: float
        alerts: int
        mmff_strain_min: float
        mmff_strain_p95: float
        mmff_strain_score_0_1: float
        ip_kcal: float
        ea_kcal: float
        hardness_kcal: float
        electrophilicity_kcal: float
        min_bde_kcal: float
        logp: float
        tpsa: float
        small_ring_count: int
        rot_bonds: int
        composite_score_0_1: float

    def _smiles_to_mol_3d(self, smiles: str, n_confs: int = 8):
        from rdkit import Chem as _Chem
        from rdkit.Chem import AllChem as _AllChem
        m0 = _Chem.MolFromSmiles(smiles)
        if m0 is None:
            raise ValueError("Bad SMILES")
        mol = _Chem.AddHs(m0)
        params = _AllChem.ETKDGv3()
        params.pruneRmsThresh = 0.5
        cids = _AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        _AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s")
        return mol, list(cids)

    def _molconf_to_ase_atoms(self, mol, cid: int):
        from ase import Atoms as _Atoms
        conf = mol.GetConformer(cid)
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        coords = conf.GetPositions()
        return _Atoms(symbols=symbols, positions=coords)

    def _mmff_strain_kcal(self, mol, cids):
        from rdkit.Chem import AllChem as _AllChem
        import numpy as _np
        res = _AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s")
        energies = {cid: e[1] for cid, e in zip(cids, res)}
        e0 = min(energies.values()) if energies else 0.0
        return {cid: energies[cid] - e0 for cid in cids}

    def _ersatz_sa_score(self, mol) -> float:
        from rdkit.Chem import rdMolDescriptors as _rdd
        rings = _rdd.CalcNumRings(mol)
        spiro = _rdd.CalcNumSpiroAtoms(mol)
        bridged = _rdd.CalcNumBridgeheadAtoms(mol)
        size = mol.GetNumAtoms()
        heteros = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        return min(10.0, 1.5 + 0.03 * size + 0.6 * rings + 0.8 * spiro + 0.8 * bridged + 0.2 * heteros)

    def _count_alerts(self, mol) -> int:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            fc = FilterCatalog(params)
            return sum(1 for _ in fc.GetMatches(mol))
        except Exception:
            return 0

    def _xtb_single_point_kcal(self, atoms, charge=0, uhf=0) -> float:
        from tblite.ase import TBLite as _TBLite
        EV2KCAL = 23.060549
        calc = _TBLite(method="GFN2-xTB", charge=charge, uhf=uhf)
        atoms.calc = calc
        e_eV = atoms.get_potential_energy()
        return float(e_eV * EV2KCAL)

    def _ip_ea_kcal(self, atoms):
        E0 = self._xtb_single_point_kcal(atoms, charge=0, uhf=0)
        Ec = self._xtb_single_point_kcal(atoms, charge=+1, uhf=1)
        Ea = self._xtb_single_point_kcal(atoms, charge=-1, uhf=1)
        IP = Ec - E0
        EA = E0 - Ea
        return IP, EA

    def _hardness_electrophilicity(self, IP_kcal: float, EA_kcal: float):
        mu = -(IP_kcal + EA_kcal) / 2.0
        eta = max(1e-6, (IP_kcal - EA_kcal) / 2.0)
        omega = (mu * mu) / (2.0 * eta)
        return mu, eta, omega

    def _candidate_bonds(self, mol):
        from rdkit import Chem as _Chem
        bonds = []
        for b in mol.GetBonds():
            if b.GetBondType() != _Chem.BondType.SINGLE:
                continue
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
                continue
            bonds.append((a1.GetIdx(), a2.GetIdx()))
        return bonds

    def _bde_vertical_kcal(self, mol, atoms, bond):
        from rdkit import Chem as _Chem
        from rdkit.Chem import AllChem as _AllChem
        from ase import Atoms as _Atoms
        i, j = bond
        if not mol.GetBondBetweenAtoms(i, j):
            return float('inf')
        emol = _Chem.EditableMol(_Chem.RemoveHs(mol))
        emol.RemoveBond(i, j)
        frag = emol.GetMol()
        frags = _Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
        if len(frags) != 2:
            return float('inf')
        fragsH = [_Chem.AddHs(f) for f in frags]
        frag_atoms = []
        for f in fragsH:
            try:
                _AllChem.EmbedMolecule(f, _AllChem.ETKDGv3())
                _AllChem.MMFFOptimizeMolecule(f, mmffVariant="MMFF94s")
            except Exception:
                pass
            conf = f.GetConformer()
            sym = [a.GetSymbol() for a in f.GetAtoms()]
            pos = conf.GetPositions()
            frag_atoms.append(_Atoms(symbols=sym, positions=pos))
        E_AB = self._xtb_single_point_kcal(atoms, charge=0, uhf=0)
        E_rad_sum = sum(self._xtb_single_point_kcal(a, charge=0, uhf=1) for a in frag_atoms)
        return E_rad_sum - E_AB

    def _feasibility_from_smiles(self, smiles: str, temperature_k: float = 298.15, n_confs: int = 8):
        from rdkit.Chem import Descriptors as _Descriptors
        from rdkit.Chem import Crippen as _Crippen
        from rdkit.Chem import rdMolDescriptors as _rdd
        import numpy as _np
        try:
            mol, cids = self._smiles_to_mol_3d(smiles, n_confs)
        except Exception:
            return None
        sa = self._ersatz_sa_score(mol)
        alerts = self._count_alerts(mol)
        logp = _Crippen.MolLogP(mol)
        tpsa = _rdd.CalcTPSA(mol)
        ring_info = mol.GetRingInfo()
        small_rings = sum(1 for r in ring_info.AtomRings() if len(r) in (3, 4))
        rotb = _Descriptors.NumRotatableBonds(mol)
        strain = self._mmff_strain_kcal(mol, cids)
        strains = [strain[c] for c in cids] if cids else [0.0]
        strain_min = float(min(strains))
        strain_p95 = float(_np.percentile(strains, 95)) if len(strains) > 1 else strain_min
        try:
            atoms0 = self._molconf_to_ase_atoms(mol, cids[0] if cids else mol.GetConformer().GetId())
        except Exception:
            atoms0 = None
        IP, EA = 0.0, 0.0
        if atoms0 is not None:
            try:
                IP, EA = self._ip_ea_kcal(atoms0)
            except Exception:
                IP, EA = 0.0, 0.0
        mu, eta, omega = self._hardness_electrophilicity(IP, EA)
        bonds = self._candidate_bonds(mol)[:8]
        min_bde = float('inf')
        if atoms0 is not None:
            for b in bonds:
                try:
                    bde = self._bde_vertical_kcal(mol, atoms0, b)
                    if bde < min_bde:
                        min_bde = bde
                except Exception:
                    continue
        if min_bde == float('inf'):
            min_bde = 0.0
        s = 1.0
        s -= 0.06 * max(0, sa - 3.0)
        s -= 0.15 * min(alerts, 2)
        s -= 0.02 * max(0.0, strain_p95 - 2.0)
        s -= 0.04 * max(0.0, (5.0 - min_bde) / 5.0)
        s -= 0.02 * small_rings
        s -= 0.01 * max(0, rotb - 8)
        s += 0.005 * min(20.0, eta) / 10.0
        s = max(0.0, min(1.0, s))
        # MMFF-only feasibility (0..1) from strain (use same baseline and slope as composite penalty)
        strain_only_feas = max(0.0, min(1.0, 1.0 - 0.02 * max(0.0, strain_p95 - 2.0)))
        return self.FeasibilityReport(
            sa_score=round(sa, 2),
            alerts=int(alerts),
            mmff_strain_min=round(strain_min, 2),
            mmff_strain_p95=round(strain_p95, 2),
            mmff_strain_score_0_1=round(strain_only_feas, 3),
            ip_kcal=round(IP, 2),
            ea_kcal=round(EA, 2),
            hardness_kcal=round(eta, 2),
            electrophilicity_kcal=round(omega, 2),
            min_bde_kcal=round(min_bde, 2),
            logp=round(logp, 2),
            tpsa=round(tpsa, 2),
            small_ring_count=int(small_rings),
            rot_bonds=int(rotb),
            composite_score_0_1=round(s, 3),
        )
    
    def calculate_fitness_score(self, properties: Dict[str, float], target_properties: Dict[str, float], 
                               weights: Dict[str, float]) -> float:
        """Calculate weighted loss: MAPE or MSE depending on self.error_metric (lower is better)."""
        weighted_sum = 0.0
        total_weight = 0.0
        epsilon = 1e-9
        use_mape = (self.error_metric == 'mape')
        
        for prop_name, target_value in target_properties.items():
            if prop_name in properties and prop_name in weights:
                current_value = float(properties[prop_name])
                target_val = float(target_value)
                weight = float(weights[prop_name])
                if use_mape:
                    denom = max(epsilon, abs(target_val))
                    err = abs(current_value - target_val) / denom
                else:
                    diff = current_value - target_val
                    err = diff * diff
                weighted_sum += weight * err
                total_weight += weight
        
        return (weighted_sum / total_weight) if total_weight > 0 else weighted_sum

    def calculate_combined_score(self, property_error: float, feasibility: Optional['MolecularOptimizationAgent.FeasibilityReport']) -> float:
        """Return property error only (MAPE/MSE). Feasibility is enforced via threshold filter elsewhere."""
        return float(property_error)

    def _feasibility_to_public_dict(self, feas: Optional['MolecularOptimizationAgent.FeasibilityReport']) -> Optional[Dict[str, Any]]:
        """Return feasibility fields for reporting, including composite_score_0_1."""
        if feas is None:
            return None
        d = dict(feas.__dict__)
        return d
    
    def run_beam_search_optimization(self, starting_molecule: str, target_properties: Dict[str, float], 
                                   weights: Dict[str, float], verbose: bool = True, cancel_event: Any = None) -> Dict[str, Any]:
        """Run beam search optimization with LangGraph integration"""
        
        # Initialize optimization state
        state = OptimizationState(
            current_molecule=starting_molecule,
            target_properties=target_properties,
            weights=weights,
            beam_candidates=[],
            iteration=0,
            max_iterations=self.max_iterations,
            beam_width=self.beam_width,
            best_score=0.0,
            best_molecule=starting_molecule,
            best_gibbs=None,
            search_history=[],
            convergence_threshold=self.convergence_threshold,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print("MOLECULAR OPTIMIZATION AGENT - BEAM SEARCH")
            print(f"{'='*60}")
            print(f"Starting molecule: {starting_molecule}")
            print(f"Target properties: {target_properties}")
            print(f"Weights: {weights}")
            print(f"Beam width: {self.beam_width}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"RAG enabled: {self.use_rag}")
        
        # Get initial molecule properties
        initial_properties = predict_properties.invoke(starting_molecule)
        initial_prop_error = self.calculate_fitness_score(initial_properties, target_properties, weights)
        initial_feas = self._feasibility_from_smiles(starting_molecule)
        initial_score = self.calculate_combined_score(initial_prop_error, initial_feas)
        # Enforce feasibility threshold
        initial_feas_score = float(getattr(initial_feas, 'composite_score_0_1', 0.0)) if initial_feas else 0.0
        if initial_feas_score < self.feasibility_threshold:
            if verbose:
                print(f"Initial molecule filtered out by feasibility < {self.feasibility_threshold:.2f}; using fallback TNT")
            starting_molecule = 'Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]'
            initial_properties = predict_properties.invoke(starting_molecule)
            initial_prop_error = self.calculate_fitness_score(initial_properties, target_properties, weights)
            initial_feas = self._feasibility_from_smiles(starting_molecule)
            initial_score = self.calculate_combined_score(initial_prop_error, initial_feas)
        state.best_score = initial_score
        
        if verbose:
            print(f"\nInitial molecule score: {initial_score:.4f}")
            print(f"Initial properties: {initial_properties}")
        
        # Initialize beam with starting molecule and assign ordinal ID
        next_id = 1
        beam = [{
            'id': next_id,
            'smiles': starting_molecule,
            'properties': initial_properties,
            'score': initial_score,
            'prop_error': initial_prop_error,
            'feasibility': self._feasibility_to_public_dict(initial_feas),
            'feasibility_score': float(getattr(initial_feas, 'composite_score_0_1', getattr(initial_feas, 'mmff_strain_score_0_1', 0.0))) if initial_feas else 0.0,
            'parent': None,
            'modification': 'Initial molecule',
            'iteration': 0
        }]
        next_id += 1
        
        visited_smiles = {starting_molecule}
        previous_best_score = state.best_score
        no_improvement_iterations = 0
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Cooperative cancellation
            try:
                if cancel_event is not None and getattr(cancel_event, 'is_set', lambda: False)():
                    if verbose:
                        print("\nOptimization cancelled by user")
                    break
            except Exception:
                pass
            state.iteration = iteration
            
            if verbose:
                print(f"\n{'='*40}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*40}")
                print(f"Current beam size: {len(beam)}")
            
            # Generate candidates from current beam as a tree: each parent yields exactly beam_width children
            all_candidates = []
            
            for candidate in beam:
                if verbose:
                    print(f"\n  Modifying [ID {candidate.get('id','?')}]: {candidate['smiles']} (score: {candidate['score']:.4f})")
                
                # Use LangGraph and RAG to generate modifications
                # Cancellation check per parent
                try:
                    if cancel_event is not None and getattr(cancel_event, 'is_set', lambda: False)():
                        if verbose:
                            print("    Cancellation requested; stopping candidate generation")
                        break
                except Exception:
                    pass
                modifications = self._get_modifications_with_agent(candidate['smiles'], target_properties, candidate['score'], verbose)
                
                parent_candidates = []
                for modification in modifications:
                    modified_smiles = modification.get('smiles')
                    if not modified_smiles or modified_smiles in visited_smiles:
                        continue
                    if '.' in modified_smiles:
                        if verbose:
                            print("    -> Skipping multi-component SMILES candidate")
                        continue
                    # Validate and score
                    validation = validate_molecule_structure.invoke(modified_smiles)
                    if not validation.get('valid', False):
                        if verbose:
                            print(f"    -> Invalid modification: {validation.get('error', 'Unknown error')}")
                        continue
                    properties = predict_properties.invoke(modified_smiles)
                    prop_error = self.calculate_fitness_score(properties, target_properties, weights)
                    feas = self._feasibility_from_smiles(modified_smiles)
                    feas_ok = (float(getattr(feas, 'composite_score_0_1', 0.0)) >= self.feasibility_threshold) if feas else False
                    if not feas_ok:
                        if verbose:
                            print("    -> Filtered by feasibility threshold")
                        continue
                    score = self.calculate_combined_score(prop_error, feas)
                    parent_candidates.append({
                        'smiles': modified_smiles,
                        'properties': properties,
                        'score': score,
                        'prop_error': prop_error,
                        'feasibility': self._feasibility_to_public_dict(feas),
                        'feasibility_score': float(getattr(feas, 'composite_score_0_1', getattr(feas, 'mmff_strain_score_0_1', 0.0))) if feas else 0.0,
                        'parent': candidate['smiles'],
                        'modification': modification.get('description', 'Unknown modification'),
                        'iteration': iteration + 1,
                        'validation': validation
                    })
                
                # Exhaustively top-up candidates for this parent to reach beam_width
                if len(parent_candidates) < self.beam_width:
                    needed = self.beam_width - len(parent_candidates)
                    extra_mods = generate_molecular_modifications.invoke(candidate['smiles'], max_modifications=max(self.beam_width * 10, needed))
                    for mod in extra_mods:
                        if len(parent_candidates) >= self.beam_width:
                            break
                        mod_smiles = mod.get('modified_smiles') or mod.get('smiles')
                        if not mod_smiles or mod_smiles in visited_smiles:
                            continue
                        if '.' in mod_smiles:
                            continue
                        validation = validate_molecule_structure.invoke(mod_smiles)
                        if not validation.get('valid', False):
                            continue
                        properties = predict_properties.invoke(mod_smiles)
                        prop_error = self.calculate_fitness_score(properties, target_properties, weights)
                        feas = self._feasibility_from_smiles(mod_smiles)
                        feas_ok = (float(getattr(feas, 'composite_score_0_1', 0.0)) >= self.feasibility_threshold) if feas else False
                        if not feas_ok:
                            continue
                        score = self.calculate_combined_score(prop_error, feas)
                        # Deduplicate within parent_candidates
                        if any(pc['smiles'] == mod_smiles for pc in parent_candidates):
                            continue
                        parent_candidates.append({
                            'smiles': mod_smiles,
                            'properties': properties,
                            'score': score,
                            'prop_error': prop_error,
                            'feasibility': self._feasibility_to_public_dict(feas),
                            'feasibility_score': float(getattr(feas, 'composite_score_0_1', getattr(feas, 'mmff_strain_score_0_1', 0.0))) if feas else 0.0,
                            'parent': candidate['smiles'],
                            'modification': mod.get('description', 'Generated modification'),
                            'iteration': iteration + 1,
                            'validation': validation
                        })
                # Select up to beam_width (minimize overall MSE)
                def _pc_key(c: Dict[str, Any]):
                    # Primary sort by combined score (lower better)
                    return (float(c.get('score', 1e12)),)
                parent_candidates.sort(key=_pc_key)
                selected_children = parent_candidates[:min(self.beam_width, len(parent_candidates))]
                for child in selected_children:
                    child['id'] = next_id
                    next_id += 1
                    all_candidates.append(child)
                    visited_smiles.add(child['smiles'])
                    if verbose:
                        # Verbose both components
                        pe = float(child.get('prop_error', float('nan')))
                        fs = float(child.get('feasibility_score', 0.0))
                        print(f"    -> SMILES: {child['smiles']} | modification: {child['modification']} | score (combined): {child['score']:.4f} | prop_err: {pe:.4f} | feas: {fs:.3f}")
                    # Update best (minimize combined)
                    if child['score'] < state.best_score:
                        state.best_score = child['score']
                        state.best_molecule = child['smiles']
                        if verbose:
                            print(f"    *** NEW BEST (MAPE): {child['score']:.4f} ***")
            
            # Backfill across parents to reach at least proceed_k total candidates, if possible
            if len(all_candidates) < self.proceed_k:
                needed_total = self.proceed_k - len(all_candidates)
                parents_sorted = sorted(beam, key=lambda x: x['score'])
                existing_smiles = {c['smiles'] for c in all_candidates} | visited_smiles
                for parent in parents_sorted:
                    if needed_total <= 0:
                        break
                    try:
                        extra_mods = generate_molecular_modifications.invoke(parent['smiles'], max_modifications=max(self.beam_width * 20, needed_total))
                    except Exception:
                        extra_mods = []
                    for mod in extra_mods:
                        if needed_total <= 0:
                            break
                        mod_smiles = mod.get('modified_smiles') or mod.get('smiles')
                        if not mod_smiles or mod_smiles in existing_smiles:
                            continue
                        if '.' in mod_smiles:
                            continue
                        validation = validate_molecule_structure.invoke(mod_smiles)
                        if not validation.get('valid', False):
                            continue
                        properties = predict_properties.invoke(mod_smiles)
                        prop_error = self.calculate_fitness_score(properties, target_properties, weights)
                        feas = self._feasibility_from_smiles(mod_smiles)
                        feas_ok = (float(getattr(feas, 'composite_score_0_1', 0.0)) >= self.feasibility_threshold) if feas else False
                        if not feas_ok:
                            continue
                        score = self.calculate_combined_score(prop_error, feas)
                        child = {
                            'smiles': mod_smiles,
                            'properties': properties,
                            'score': score,
                            'prop_error': prop_error,
                            'feasibility': self._feasibility_to_public_dict(feas),
                            'feasibility_score': float(getattr(feas, 'composite_score_0_1', getattr(feas, 'mmff_strain_score_0_1', 0.0))) if feas else 0.0,
                            'parent': parent['smiles'],
                            'modification': mod.get('description', 'Generated modification'),
                            'iteration': iteration + 1,
                            'validation': validation
                        }
                        child['id'] = next_id
                        next_id += 1
                        all_candidates.append(child)
                        existing_smiles.add(mod_smiles)
                        visited_smiles.add(mod_smiles)
                        needed_total -= 1
                        if verbose:
                            print(f"    -> SMILES: {child['smiles']} | modification: {child['modification']} | score (MAPE): {child['score']:.4f}")

            # Last-resort exhaustive fallback: accept any RDKit-parseable modifications if still empty
            if len(all_candidates) == 0:
                if verbose:
                    print("    No valid candidates after standard generation; attempting exhaustive fallback...")
                from rdkit import Chem as _Chem
                parents_sorted = sorted(beam, key=lambda x: x['score'])
                for parent in parents_sorted:
                    try:
                        raw_mods = generate_molecular_modifications.invoke(parent['smiles'], max_modifications=max(self.beam_width * 50, self.proceed_k * 20))
                    except Exception:
                        raw_mods = []
                    for mod in raw_mods:
                        if len(all_candidates) >= self.proceed_k:
                            break
                        mod_smiles = mod.get('modified_smiles') or mod.get('smiles')
                        if not mod_smiles or mod_smiles in visited_smiles:
                            continue
                        if '.' in mod_smiles:
                            continue
                        try:
                            mol_ok = _Chem.MolFromSmiles(mod_smiles) is not None
                        except Exception:
                            mol_ok = False
                        if not mol_ok:
                            continue
                        # Compute properties without strict structural validation
                        properties = predict_properties.invoke(mod_smiles)
                        prop_error = self.calculate_fitness_score(properties, target_properties, weights)
                        feas = self._feasibility_from_smiles(mod_smiles)
                        score = self.calculate_combined_score(prop_error, feas)
                        child = {
                            'smiles': mod_smiles,
                            'properties': properties,
                            'score': score,
                            'prop_error': prop_error,
                            'feasibility': self._feasibility_to_public_dict(feas),
                            'feasibility_score': float(getattr(feas, 'composite_score_0_1', getattr(feas, 'mmff_strain_score_0_1', 0.0))) if feas else 0.0,
                            'parent': parent['smiles'],
                            'modification': mod.get('description', 'Exhaustive fallback modification'),
                            'iteration': iteration + 1,
                            'validation': {'valid': True}
                        }
                        child['id'] = next_id
                        next_id += 1
                        all_candidates.append(child)
                        visited_smiles.add(mod_smiles)
                        if verbose:
                            print(f"    -> [Fallback] SMILES: {child['smiles']} | score (MAPE): {child['score']:.4f}")
                    if len(all_candidates) >= self.proceed_k:
                        break
                # Absolute last-ditch: apply simple hydrogen substitution if still empty
                if len(all_candidates) == 0:
                    for parent in parents_sorted:
                        try:
                            simple = self._substitute_hydrogen(parent['smiles'])
                        except Exception:
                            simple = None
                        if simple and simple not in visited_smiles:
                            properties = predict_properties.invoke(simple)
                            prop_error = self.calculate_fitness_score(properties, target_properties, weights)
                            feas = self._feasibility_from_smiles(simple)
                            feas_ok = (float(getattr(feas, 'composite_score_0_1', 0.0)) >= self.feasibility_threshold) if feas else False
                            if not feas_ok:
                                continue
                            score = self.calculate_combined_score(prop_error, feas)
                            child = {
                                'smiles': simple,
                                'properties': properties,
                                'score': score,
                                'prop_error': prop_error,
                                'feasibility': self._feasibility_to_public_dict(feas),
                                'feasibility_score': float(getattr(feas, 'composite_score_0_1', getattr(feas, 'mmff_strain_score_0_1', 0.0))) if feas else 0.0,
                                'parent': parent['smiles'],
                                'modification': 'Simple hydrogen substitution (fallback)',
                                'iteration': iteration + 1,
                                'validation': {'valid': True}
                            }
                            child['id'] = next_id
                            next_id += 1
                            all_candidates.append(child)
                            visited_smiles.add(simple)
                            if verbose:
                                print(f"    -> [Fallback] SMILES: {child['smiles']} | score (MAPE): {child['score']:.4f}")
                            if len(all_candidates) >= self.proceed_k:
                                break
            
            # Select top candidates for next beam (minimize combined score)
            def _beam_key(c: Dict[str, Any]):
                return (float(c.get('score', 1e12)),)
            all_candidates.sort(key=_beam_key)
            beam = all_candidates[:self.proceed_k]
            if verbose and len(beam) > 0:
                print(f"  Progressing to next iteration with IDs: {[c.get('id','?') for c in beam]}")
            
            # Store search history
            state.search_history.append({
                'iteration': iteration + 1,
                'beam_size': len(beam),
                'best_score': state.best_score,
                'best_molecule': state.best_molecule,
                'candidates': beam[:3]  # Store top 3 candidates
            })
            
            if verbose:
                print(f"\nBest score (MAPE) in iteration {iteration + 1}: {beam[0]['score'] if beam else 'N/A'}")
                print(f"Overall best score (MAPE): {state.best_score:.4f}")
            
            # Check for convergence (minimization) with configurable patience
            if len(beam) > 0:
                if state.best_score < previous_best_score:
                    no_improvement_iterations = 0
                    previous_best_score = state.best_score
                else:
                    no_improvement_iterations += 1
                    if self.early_stop_patience and no_improvement_iterations >= self.early_stop_patience:
                        if verbose:
                            print(f"\nConvergence reached: no improvement in {no_improvement_iterations} consecutive iterations")
                        break
            
            # Early stopping if no valid modifications
            if not beam:
                # If cancelled, allow graceful exit
                try:
                    if cancel_event is not None and getattr(cancel_event, 'is_set', lambda: False)():
                        break
                except Exception:
                    pass
                raise RuntimeError("No valid modifications found to proceed to next iteration")
        
        # Prepare final results
        # Collect best candidate extra metrics for display
        try:
            best_entry = None
            for entry in state.search_history[::-1]:
                for cand in entry.get('candidates', []):
                    if cand.get('smiles') == state.best_molecule:
                        best_entry = cand
                        break
                if best_entry:
                    break
            best_prop_error = float(best_entry.get('prop_error')) if best_entry and 'prop_error' in best_entry else None
            best_feasibility_score = float(best_entry.get('feasibility_score')) if best_entry and 'feasibility_score' in best_entry else None
        except Exception:
            best_prop_error = None
            best_feasibility_score = None

        results = {
            'starting_molecule': starting_molecule,
            'target_properties': target_properties,
            'weights': weights,
            'best_molecule': state.best_molecule,
            'best_score': state.best_score,
            # keep key for backward-compatibility but now None
            'best_gibbs_kcal_mol': None,
            'best_prop_error': best_prop_error,
            'best_feasibility_score': best_feasibility_score,
            'best_properties': predict_properties.invoke(state.best_molecule) if state.best_molecule != starting_molecule else initial_properties,
            'total_iterations': iteration + 1,
            'search_history': state.search_history,
            'visited_molecules': len(visited_smiles)
        }
        
        return results
    
    def _get_modifications_with_agent(self, smiles: str, target_properties: Dict[str, float], current_score: float, verbose: bool = True) -> List[Dict[str, Any]]:
        """Use the LangGraph agent and RAG to generate molecular modifications"""
        # If RAG is disabled, use tool-based generator directly
        if not self.use_rag:
            return generate_molecular_modifications.invoke(smiles, max_modifications=10)

        if verbose:
            print(f"    Generating modifications for: {smiles}")
        
        # First, try to find modifications using RAG
        rag_modifications = self._find_modifications_with_rag(smiles, target_properties, current_score)
        
        if verbose:
            print(f"    Found {len(rag_modifications)} RAG modifications")
        
        # Then use the agent to generate additional modifications (with timeout)
        agent_modifications = self._generate_modifications_with_agent(smiles, target_properties)
        
        if verbose:
            print(f"    Found {len(agent_modifications)} agent modifications")
        
        # Combine and return unique modifications
        all_modifications = rag_modifications + agent_modifications
        
        # Remove duplicates based on SMILES
        unique_modifications = []
        seen_smiles = set()
        for mod in all_modifications:
            if mod['smiles'] not in seen_smiles:
                unique_modifications.append(mod)
                seen_smiles.add(mod['smiles'])
        
        if verbose:
            print(f"    Total unique modifications: {len(unique_modifications)}")
        
        return unique_modifications
    
    def _find_modifications_with_rag(self, smiles: str, target_properties: Dict[str, float], current_score: float) -> List[Dict[str, Any]]:
        """Find molecular modifications using RAG"""
        modifications = []
        
        try:
            # Get current molecule properties
            current_properties = predict_properties.invoke(smiles)
            
            # Create search queries for modifications
            search_queries = self._generate_modification_queries(smiles, current_properties, target_properties)
            
            for i, query in enumerate(search_queries[:3]):  # Limit to first 3 queries to avoid hanging
                try:
                    # Search RAG for modification strategies
                    rag_results = retrieve_context.invoke(query)
                    
                    if rag_results and len(rag_results) > 0:
                        # Extract modification information from RAG results
                        rag_modifications = self._extract_modifications_from_rag(rag_results, smiles)
                        modifications.extend(rag_modifications)
                        
                except Exception as e:
                    continue
            
            return modifications
            
        except Exception as e:
            return modifications
    
    def _generate_modification_queries(self, smiles: str, current_properties: Dict[str, float], target_properties: Dict[str, float]) -> List[str]:
        """Generate search queries for finding modifications"""
        queries = []
        
        # Analyze what properties need improvement
        for prop_name, target_value in target_properties.items():
            if prop_name in current_properties:
                current_value = current_properties[prop_name]
                
                if prop_name == "Density" and current_value < target_value:
                    queries.extend([
                        "increase density energetic materials molecular modifications",
                        "high density substituents energetic compounds",
                        "density enhancement energetic materials"
                    ])
                
                elif prop_name == "Detonation velocity" and current_value < target_value:
                    queries.extend([
                        "increase detonation velocity molecular modifications",
                        "high detonation velocity substituents",
                        "detonation velocity enhancement energetic materials"
                    ])
                
                elif prop_name == "Explosion pressure" and current_value < target_value:
                    queries.extend([
                        "increase explosion pressure molecular modifications",
                        "high explosion pressure substituents",
                        "explosion pressure enhancement energetic materials"
                    ])
        
        # Add general modification queries
        queries.extend([
            "molecular modifications energetic materials",
            "functional group addition energetic compounds",
            "nitro group addition energetic materials",
            "azido group energetic materials",
            "nitramine synthesis energetic materials",
            "tetrazole synthesis energetic materials",
            "triazole energetic materials",
            "furazan energetic materials"
        ])
        
        return queries
    
    def _extract_modifications_from_rag(self, rag_results: List[Dict[str, Any]], current_smiles: str) -> List[Dict[str, Any]]:
        """Extract modification strategies from RAG results"""
        modifications = []
        
        for result in rag_results:
            content = result.get('Content', '')
            title = result.get('Title', '')
            authors = result.get('Authors', 'Unknown Authors')
            year = result.get('Year', '')
            
            # Look for modification patterns in the content
            import re
            
            # Common modification patterns
            modification_patterns = [
                (r'add\s+nitro\s+group', 'nitro_addition'),
                (r'add\s+azido\s+group', 'azido_addition'),
                (r'add\s+nitramine', 'nitramine_addition'),
                (r'add\s+tetrazole', 'tetrazole_addition'),
                (r'add\s+triazole', 'triazole_addition'),
                (r'add\s+furazan', 'furazan_addition'),
                (r'remove\s+nitro\s+group', 'nitro_removal'),
                (r'remove\s+hydroxyl', 'hydroxyl_removal'),
                (r'remove\s+amino', 'amino_removal'),
                (r'substitute\s+hydrogen', 'hydrogen_substitution'),
                (r'replace\s+carbon\s+with\s+nitrogen', 'carbon_nitrogen_substitution'),
            ]
            
            for pattern, mod_type in modification_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Try to apply this modification
                    modified_smiles = self._apply_modification_from_rag(current_smiles, mod_type)
                    if modified_smiles and modified_smiles != current_smiles:
                        modifications.append({
                            'smiles': modified_smiles,
                            'description': f'RAG-suggested {mod_type}',
                            'source': title,
                            'authors': authors,
                            'year': year,
                            'rag_content': content[:100] + "..."
                        })
        
        return modifications
    
    def _apply_modification_from_rag(self, smiles: str, modification_type: str) -> Optional[str]:
        """Apply a modification based on RAG suggestion"""
        try:
            if modification_type == 'nitro_addition':
                return self._add_nitro_group(smiles)
            elif modification_type == 'azido_addition':
                return self._add_azido_group(smiles)
            elif modification_type == 'nitramine_addition':
                return self._add_nitramine_group(smiles)
            elif modification_type == 'tetrazole_addition':
                return self._add_tetrazole_group(smiles)
            elif modification_type == 'nitro_removal':
                return self._remove_nitro_group(smiles)
            elif modification_type == 'hydrogen_substitution':
                return self._substitute_hydrogen(smiles)
            else:
                return None
        except:
            return None
    
    def _add_nitro_group(self, smiles: str) -> Optional[str]:
        """Add a nitro group to the molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon atom with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add nitro group
                            n_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            o1_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            o2_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            
                            # Add bonds
                            modified_mol.AddBond(atom.GetIdx(), n_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n_atom, o1_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n_atom, o2_atom, Chem.BondType.SINGLE)
                            
                            # Set formal charges
                            modified_mol.GetAtomWithIdx(n_atom).SetFormalCharge(1)
                            modified_mol.GetAtomWithIdx(o2_atom).SetFormalCharge(-1)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def _add_azido_group(self, smiles: str) -> Optional[str]:
        """Add an azido group to the molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon atom with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add azido group
                            n1_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n2_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n3_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            
                            # Add bonds
                            modified_mol.AddBond(atom.GetIdx(), n1_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n1_atom, n2_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n2_atom, n3_atom, Chem.BondType.TRIPLE)
                            
                            # Set formal charges
                            modified_mol.GetAtomWithIdx(n1_atom).SetFormalCharge(-1)
                            modified_mol.GetAtomWithIdx(n2_atom).SetFormalCharge(1)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def _add_nitramine_group(self, smiles: str) -> Optional[str]:
        """Add a nitramine group to the molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a nitrogen atom with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add nitro group to nitrogen
                            n_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            o1_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            o2_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            
                            # Add bonds
                            modified_mol.AddBond(atom.GetIdx(), n_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n_atom, o1_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n_atom, o2_atom, Chem.BondType.SINGLE)
                            
                            # Set formal charges
                            modified_mol.GetAtomWithIdx(n_atom).SetFormalCharge(1)
                            modified_mol.GetAtomWithIdx(o2_atom).SetFormalCharge(-1)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def _add_tetrazole_group(self, smiles: str) -> Optional[str]:
        """Add a tetrazole group to the molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon atom with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add tetrazole ring
                            n1_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n2_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n3_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n4_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            c_atom = modified_mol.AddAtom(Chem.Atom('C'))
                            
                            # Add bonds to form tetrazole ring
                            modified_mol.AddBond(atom.GetIdx(), n1_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n1_atom, n2_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n2_atom, n3_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n3_atom, n4_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n4_atom, c_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(c_atom, n1_atom, Chem.BondType.SINGLE)
                            
                            # Add hydrogen to carbon
                            h_atom = modified_mol.AddAtom(Chem.Atom('H'))
                            modified_mol.AddBond(c_atom, h_atom, Chem.BondType.SINGLE)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def _remove_nitro_group(self, smiles: str) -> Optional[str]:
        """Remove a nitro group from the molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find nitro group pattern
            nitro_pattern = Chem.MolFromSmarts('[#6][#7+](=[#8])[#8-]')
            matches = mol.GetSubstructMatches(nitro_pattern)
            
            for match in matches:
                try:
                    modified_mol = Chem.RWMol(mol)
                    
                    # Remove nitro group atoms (in reverse order)
                    atoms_to_remove = sorted(match[1:], reverse=True)  # Keep the carbon
                    for atom_idx in atoms_to_remove:
                        modified_mol.RemoveAtom(atom_idx)
                    
                    # Add hydrogen to the carbon
                    carbon_idx = match[0]
                    h_atom = modified_mol.AddAtom(Chem.Atom('H'))
                    modified_mol.AddBond(carbon_idx, h_atom, Chem.BondType.SINGLE)
                    
                    Chem.SanitizeMol(modified_mol)
                    return Chem.MolToSmiles(modified_mol)
                    
                except:
                    continue
            
            return None
        except:
            return None
    
    def _substitute_hydrogen(self, smiles: str) -> Optional[str]:
        """Substitute hydrogen with a more energetic group"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add nitro group (most common substitution)
                            n_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            o1_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            o2_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            
                            # Add bonds
                            modified_mol.AddBond(atom.GetIdx(), n_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n_atom, o1_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n_atom, o2_atom, Chem.BondType.SINGLE)
                            
                            # Set formal charges
                            modified_mol.GetAtomWithIdx(n_atom).SetFormalCharge(1)
                            modified_mol.GetAtomWithIdx(o2_atom).SetFormalCharge(-1)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def _generate_modifications_with_agent(self, smiles: str, target_properties: Dict[str, float]) -> List[Dict[str, Any]]:
        """Use the LangGraph agent to generate molecular modifications"""
        
        prompt = f"""
        You are an expert in energetic materials molecular design. Given the molecule with SMILES: {smiles}
        and target properties: {target_properties}
        
        Generate molecular modifications that could improve its energetic properties. Consider both:
        
        ADDITION strategies:
        1. Adding energetic functional groups (nitro, azido, nitramine, etc.)
        2. Ring modifications to create energetic heterocycles
        3. Substituent additions that enhance energetic performance
        
        REMOVAL strategies:
        1. Removing problematic substituents that reduce stability
        2. Removing terminal atoms that don't contribute to energetic properties
        3. Simplifying complex substituents to improve performance
        4. Removing groups that may cause instability or poor properties
        
        Use the available tools to:
        1. First validate the current molecule structure
        2. Calculate its current molecular descriptors
        3. Check for existing energetic functional groups
        4. Generate valid modifications (both additions and removals)
        5. Validate each modification
        
        Return only valid, chemically reasonable modifications that would be suitable for energetic materials.
        Consider that sometimes removing groups can be as beneficial as adding them for optimizing energetic properties.
        """
        
        try:
            # Run the LangGraph workflow with timeout using threading
            import threading
            import queue
            import time
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def run_agent():
                try:
                    states = self.app.invoke({
                        'messages': [HumanMessage(content=prompt)]
                    }, config={'configurable': {'thread_id': f'mod_{smiles[:20]}'}})
                    result_queue.put(states)
                except Exception as e:
                    exception_queue.put(e)
            
            # Start agent in separate thread
            agent_thread = threading.Thread(target=run_agent)
            agent_thread.daemon = True
            agent_thread.start()
            
            # Wait for result with timeout
            try:
                states = result_queue.get(timeout=30)  # 30 second timeout
            except queue.Empty:
                raise TimeoutError("Agent timed out while generating modifications")
            
            # Check for exceptions
            try:
                exception = exception_queue.get_nowait()
                raise exception
            except queue.Empty:
                pass  # No exception
            
            # Extract modifications from the response with strict SMILES parsing and RDKit validation
            import re
            modifications: List[Dict[str, Any]] = []
            seen_local: set = set()
            strict_pattern = r'(?<![A-Za-z0-9])(?=[^\s]*[\[\]()0-9=#@+\-\\/])(?:\[[^\]]+\]|Br|Cl|[BCNOFPSI][lr]?|[A-Z][a-z]?|[=#@+\-]|\\|/|\(|\)|\d)+'
            for message in states['messages']:
                text = getattr(message, 'content', '') or ''
                if not text:
                    continue
                candidates = re.findall(strict_pattern, text)
                for cand in candidates:
                    cand = cand.strip().strip('.,;:')
                    if not cand or cand == smiles:
                        continue
                    # Skip multi-component suggestions
                    if '.' in cand:
                        continue
                    if not re.search(r'(C|N|O|S|P|F|Cl|Br|I|c|n|o|s)', cand):
                        continue
                    try:
                        mol = Chem.MolFromSmiles(cand)
                    except Exception:
                        mol = None
                    if mol is None:
                        continue
                    # Optional sanitize to weed out invalid constructs
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        continue
                    if cand in seen_local:
                        continue
                    seen_local.add(cand)
                    # Add validation for parity with tool-generated modifications
                    try:
                        validation = validate_molecule_structure.invoke(cand)
                    except Exception:
                        validation = {'valid': True}
                    modifications.append({
                        'smiles': cand,
                        'modified_smiles': cand,
                        'description': f'Agent-generated modification: {cand}',
                        'validation': validation
                    })

            if not modifications:
                raise RuntimeError("Agent did not generate any modifications")

            return modifications
            
        except Exception as e:
            raise RuntimeError(f"Error generating modifications with agent: {e}")
    
    def process_csv_input(self, csv_file_path: str, verbose: bool = True, cancel_event: Any = None, starting_smiles: Optional[str] = None, prefer_user_start: bool = False) -> Dict[str, Any]:
        """Process CSV input and run optimization without starting molecule"""
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        if len(df) == 0:
            return {"error": "CSV file is empty"}
        
        # Get first row (assuming single optimization task)
        row = df.iloc[0]
        
        # Extract target properties
        target_properties = {}
        weights = {}
        
        property_columns = ['Density', 'Detonation velocity', 'Explosion capacity', 
                           'Explosion pressure', 'Explosion heat']
        
        for prop in property_columns:
            if prop in df.columns and not pd.isna(row[prop]):
                target_properties[prop] = float(row[prop])
                # Default weight of 1.0, can be customized
                weight_col = f"{prop}_weight"
                weights[prop] = float(row[weight_col]) if weight_col in df.columns else 1.0
        
        if not target_properties:
            return {"error": "No target properties found in CSV"}
        
        # Determine starting molecule based on user preference or samples
        starting_molecule: Optional[str] = None
        if prefer_user_start and starting_smiles:
            if '.' in starting_smiles:
                return {"error": "Starting SMILES contains multiple components ('.'); provide a single-molecule SMILES."}
            try:
                mol = Chem.MolFromSmiles(starting_smiles)
            except Exception:
                mol = None
            if mol is None:
                return {"error": "Invalid starting SMILES"}
            try:
                validation = validate_molecule_structure.invoke(starting_smiles)
            except Exception:
                validation = {"valid": True}
            if not validation.get('valid', False):
                return {"error": f"Starting SMILES failed validation: {validation.get('error','invalid structure')}"}
            starting_molecule = starting_smiles
        else:
            # Choose from samples (preferred path per user request); fallback to TNT
            try:
                starting_molecule = self._find_starting_molecule_from_samples(target_properties, weights, verbose)
            except Exception:
                starting_molecule = None
            if not starting_molecule:
                starting_molecule = 'Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]'
        
        # Run optimization
        results = self.run_beam_search_optimization(starting_molecule, target_properties, weights, verbose, cancel_event=cancel_event)
        
        return results

    def _find_starting_molecule_from_samples(self, target_properties: Dict[str, float], weights: Dict[str, float], verbose: bool = True) -> Optional[str]:
        """Select a starting molecule from sample_start_molecules.csv based on target similarity.
        Expects a CSV with a 'SMILES' column and property columns matching the prediction properties.
        If property columns are missing, predicts them on-the-fly.
        """
        # Use only the sample_start_molecules.csv dataset
        p = os.path.join(os.getcwd(), 'sample_start_molecules.csv')
        sample_path = p if os.path.exists(p) else None
        if sample_path is None:
            if verbose:
                print("  Sample CSV 'sample_start_molecules.csv' not found")
            return None
        try:
            df = pd.read_csv(sample_path)
        except Exception as e:
            if verbose:
                print(f"  Failed reading {os.path.basename(sample_path)}: {e}")
            return None
        smiles_col = None
        for c in ['SMILES', 'smiles', 'Smiles']:
            if c in df.columns:
                smiles_col = c
                break
        if smiles_col is None:
            if verbose:
                print(f"  {os.path.basename(sample_path)} missing 'SMILES' column; skipping sample-based selection")
            return None
        property_columns = list(target_properties.keys())
        # Collect candidates with property score only (lower is better). Drop structural similarity.
        candidates: List[Dict[str, Any]] = []
        from rdkit import Chem as _Chem

        for _, row in df.iterrows():
            smi = str(row[smiles_col]) if not pd.isna(row[smiles_col]) else ''
            if not smi:
                continue
            # Gather properties from row if present; otherwise predict
            props: Dict[str, float] = {}
            missing = False
            for p in property_columns:
                if p in df.columns and not pd.isna(row[p]):
                    props[p] = float(row[p])
                else:
                    missing = True
            if missing:
                try:
                    props = predict_properties.invoke(smi)
                except Exception:
                    continue
            try:
                prop_score = self.calculate_fitness_score(props, target_properties, weights)
            except Exception:
                continue
            # Validate molecule structure (non-fatal)
            try:
                valid = validate_molecule_structure.invoke(smi)
                if not valid.get('valid', False):
                    continue
            except Exception:
                pass
            # Only track score; do not compute or use similarity to TNT
            candidates.append({'smiles': smi, 'prop_score': prop_score})

        if not candidates:
            return None

        # Choose the molecule with the lowest property score only
        best = min(candidates, key=lambda x: x['prop_score'])
        if verbose:
            print(f"  Using sample-based starting molecule from {os.path.basename(sample_path)}: {best['smiles']} (prop_score: {best['prop_score']:.4f})")
        return best['smiles']
    
    def _find_starting_molecule_with_rag(self, target_properties: Dict[str, float], verbose: bool = True) -> Optional[str]:
        """Find a starting molecule using RAG based on target properties"""
        
        if verbose:
            print(f"\n{'='*50}")
            print("SEARCHING FOR STARTING MOLECULE USING RAG")
            print(f"{'='*50}")
            print(f"Target properties: {target_properties}")
        
        # Create search queries based on target properties
        search_queries = self._generate_search_queries(target_properties)
        
        for query in search_queries:
            if verbose:
                print(f"\nSearching with query: {query}")
            
            try:
                # Use RAG to search for relevant molecules
                rag_results = retrieve_context.invoke(query)
                
                if rag_results and len(rag_results) > 0:
                    # Extract molecule information from RAG results
                    molecules = self._extract_molecules_from_rag(rag_results)
                    
                    for molecule_info in molecules:
                        if verbose:
                            src_title = molecule_info.get('source', 'Unknown Source')
                            src_authors = molecule_info.get('authors', 'Unknown Authors')
                            src_year = molecule_info.get('year', '')
                            year_suffix = f" ({src_year})" if src_year else ""
                            print(f"  Found molecule: {molecule_info.get('name', 'Unknown')}\n"
                                  f"    Source: {src_title}{year_suffix}\n"
                                  f"    Authors: {src_authors}")
                        
                        # Try to get SMILES for this molecule
                        smiles = self._get_smiles_from_molecule_info(molecule_info)
                        if smiles:
                            # Validate the molecule
                            validation = validate_molecule_structure.invoke(smiles)
                            if validation.get('valid', False):
                                if verbose:
                                    print(f"  Valid starting molecule found: {smiles}")
                                return smiles
                
            except Exception as e:
                if verbose:
                    print(f"  Error in RAG search: {e}")
                continue
        
        # If RAG doesn't find anything, try some common energetic materials
        fallback_molecules = [
            "CC1=CC=C(C=C1)[N+](=O)[O-]",  # TNT
            "C1=CC=C(C=C1)[N+](=O)[O-]",   # Nitrobenzene
            "C1=CC2=C(C=C1)N=N2",          # Benzotriazole
            "C1=CC2=C(C=C1)N3C=CC=C3N2",   # Imidazole
            "C1=CC=C(C=C1)C2=NNN2",        # Phenyl azide
            "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",  # RDX
            "C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",  # HMX
        ]
        
        if verbose:
            print(f"\nRAG search unsuccessful, trying fallback molecules...")
        
        for smiles in fallback_molecules:
            try:
                validation = validate_molecule_structure.invoke(smiles)
                if validation.get('valid', False):
                    if verbose:
                        print(f"  Using fallback molecule: {smiles}")
                    return smiles
            except:
                continue
        
        return None
    
    def _generate_search_queries(self, target_properties: Dict[str, float]) -> List[str]:
        """Generate search queries for RAG based on target properties"""
        queries = []
        
        # Create queries based on property ranges
        for prop_name, target_value in target_properties.items():
            if prop_name == "Density":
                if target_value > 1.8:
                    queries.append("high density energetic materials explosives")
                elif target_value > 1.5:
                    queries.append("medium density energetic materials")
                else:
                    queries.append("low density energetic materials")
            
            elif prop_name == "Detonation velocity":
                if target_value > 8000:
                    queries.append("high detonation velocity explosives RDX HMX")
                elif target_value > 6000:
                    queries.append("medium detonation velocity explosives TNT")
                else:
                    queries.append("low detonation velocity energetic materials")
            
            elif prop_name == "Explosion pressure":
                if target_value > 300:
                    queries.append("high explosion pressure energetic materials")
                elif target_value > 200:
                    queries.append("medium explosion pressure explosives")
                else:
                    queries.append("low explosion pressure energetic materials")
        
        # Add general queries
        queries.extend([
            "energetic materials explosives molecular design",
            "nitro compounds energetic materials",
            "azido compounds explosives",
            "nitramine explosives RDX HMX",
            "tetrazole energetic materials",
            "triazole explosives",
            "furazan energetic compounds"
        ])
        
        return queries
    
    def _extract_molecules_from_rag(self, rag_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract molecule information from RAG results"""
        molecules = []
        
        for result in rag_results:
            content = result.get('Content', '')
            title = result.get('Title', '')
            authors = result.get('Authors', 'Unknown Authors')
            year = result.get('Year', '')
            
            # Look for chemical names in the content
            import re
            
            # Common energetic material patterns
            patterns = [
                r'\bTNT\b', r'\bRDX\b', r'\bHMX\b', r'\bPETN\b', r'\bTATB\b',
                r'\bCL-20\b', r'\bFOX-7\b', r'\bLLM-105\b', r'\bTNAZ\b',
                r'\bNitrobenzene\b', r'\bNitrotoluene\b', r'\bDinitrotoluene\b',
                r'\bTrinitrotoluene\b', r'\bTetrazole\b', r'\bTriazole\b',
                r'\bImidazole\b', r'\bPyrazole\b', r'\bFurazan\b', r'\bOxadiazole\b'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    molecules.append({
                        'name': match,
                        'source': title,
                        'authors': authors,
                        'year': year,
                        'content': content[:200] + "..." if len(content) > 200 else content
                    })
        
        return molecules
    
    def _get_smiles_from_molecule_info(self, molecule_info: Dict[str, Any]) -> Optional[str]:
        """Get SMILES from molecule information"""
        name = molecule_info.get('name', '')
        
        if not name:
            return None
        
        try:
            # Try to convert name to SMILES
            smiles = convert_name_to_smiles.invoke(name)
            if smiles != 'Did not convert':
                return smiles
            
            # Try common energetic materials
            common_smiles = {
                'TNT': 'CC1=CC=C(C=C1)[N+](=O)[O-]',
                'RDX': 'C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]',
                'HMX': 'C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]',
                'PETN': 'C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]',
                'TATB': 'C1=CC(=CC(=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]',
                'Nitrobenzene': 'C1=CC=C(C=C1)[N+](=O)[O-]',
                'Nitrotoluene': 'CC1=CC=C(C=C1)[N+](=O)[O-]',
                'Tetrazole': 'c1nnn[nH]1',
                'Triazole': 'c1nncn1',
                'Imidazole': 'c1ncnc1',
                'Pyrazole': 'c1n[nH]cc1',
                'Furazan': 'c1n[nH]oc1',
                'Oxadiazole': 'c1n[nH]oc1'
            }
            
            return common_smiles.get(name.upper(), None)

        except:
            return None
    
    def _is_smiles(self, text: str) -> bool:
        """Check if text looks like SMILES notation"""
        # Simple heuristic: SMILES typically contains brackets, numbers, and chemical symbols
        smiles_chars = set('[](){}1234567890=#@+-.\\/')
        text_chars = set(text)
        return len(text_chars.intersection(smiles_chars)) > 0 or 'C' in text_chars

def main():
    """Main function to run the enhanced molecular optimization agent"""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Enhanced Molecular Optimization Agent")
    parser.add_argument('-r','--rag', choices=['on', 'off'], default='on', help='Enable or disable RAG (default: on)')
    parser.add_argument('-w','--beam_width', type=int, default=5, help='Beam width (default: 5)')
    parser.add_argument('-i','--max_iterations', type=int, default=8, help='Maximum number of iterations (default: 8)')
    parser.add_argument('-k','--proceed_k', type=int, default=3, help='Number of candidates to proceed each iteration (default: 3)')
    parser.add_argument('--gui', action='store_true', help='Launch GUI (Streamlit preferred, falls back to NiceGUI)')
    parser.add_argument('--metric', choices=['mape', 'mse'], default='mape', help='Error metric to optimize (default: mape)')
    args, unknown = parser.parse_known_args()
    use_rag = (args.rag.lower() == 'on')

    # Check if trained models exist
    if not os.path.exists('./trained_models/'):
        print("Trained models not found. Please run the main.py first to train the models.")
        return
    
    # Launch GUI if requested
    if args.gui:
        try:
            from streamlit_gui import main as streamlit_main
            print('Launching Streamlit GUI (run separately): streamlit run streamlit_gui.py')
        except Exception:
            try:
                from nicegui_gui import run_gui
                run_gui()
                return
            except Exception as e:
                print(f"Failed to import GUI: {e}")
                return
        # If streamlit is preferred, provide guidance instead of blocking here.
        return

    # Initialize the agent
    agent = MolecularOptimizationAgent(beam_width=args.beam_width, max_iterations=args.max_iterations, convergence_threshold=0.01, use_rag=use_rag, proceed_k=args.proceed_k, error_metric=args.metric)
    
    # Get CSV file path from user
    csv_file_path = input("Enter the path to your CSV file: ").strip()
    
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found.")
        return
    
    # Process the optimization
    print("\n" + "="*60)
    print("ENHANCED MOLECULAR OPTIMIZATION AGENT")
    print("="*60)
    
    results = agent.process_csv_input(csv_file_path, verbose=True)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Display final results
    print("\n" + "="*60)
    print("FINAL OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nStarting molecule: {results['starting_molecule']}")
    print(f"Best molecule found: {results['best_molecule']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Total molecules explored: {results['visited_molecules']}")
    
    print(f"\nTarget properties: {results['target_properties']}")
    print(f"Best molecule properties: {results['best_properties']}")
    
    # Show search history
    print(f"\nSearch History:")
    print("-" * 40)
    for history_entry in results['search_history']:
        print(f"Iteration {history_entry['iteration']}: "
              f"Best score = {history_entry['best_score']:.4f}, "
              f"Beam size = {history_entry['beam_size']}")
    
    # Save results to file
    output_file = "enhanced_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 