"""
Beam Search Orchestrator - manages the beam search algorithm.

Owns the heavy shared components (PropertyPredictor, LiteraturePropertyRetriever)
so we instantiate them ONCE per run, not once per ChemistAgent per iteration.
"""

import logging
from typing import Callable, List, Optional

from data_structures import MoleculeState, PropertyTarget
from agents.worker_agent import ChemistAgent
from config import Config
from modules.prediction import PropertyPredictor
from modules.literature_search import LiteraturePropertyRetriever

logger = logging.getLogger(__name__)


class BeamSearchEngine:
    """Orchestrates the beam search optimization process."""

    def __init__(self, config: Config, target_properties: PropertyTarget):
        self.config = config
        self.target = target_properties
        self.beam_config = config.beam_search

        self.history: List[List[MoleculeState]] = []
        self.best_ever: Optional[MoleculeState] = None

        # Shared heavy components — built once per run.
        self.predictor = PropertyPredictor(config.system.models_directory)

        self.literature_retriever: Optional[LiteraturePropertyRetriever] = None
        if config.literature.enable_literature_search:
            self.literature_retriever = LiteraturePropertyRetriever(
                use_llm=config.literature.use_llm,
                max_papers=config.literature.max_papers,
                timeout=config.literature.timeout,
                openai_api_key=config.literature.openai_api_key,
                cache_path=config.literature.cache_path,
                ollama_base_url=getattr(config.literature, 'ollama_base_url', None),
                ollama_model=getattr(config.literature, 'ollama_model', 'ALIENTELLIGENCE/chemicalengineer'),
            )
            logger.info("BeamSearchEngine: shared literature retriever initialized")

        # Optional observer hooks for GUI / external callers.
        # Signature: on_iteration(iteration, all_candidates, beam)
        self.on_seed: Optional[Callable[[MoleculeState], None]] = None
        self.on_iteration: Optional[Callable[[int, List[MoleculeState], List[MoleculeState]], None]] = None
        self.on_best: Optional[Callable[[MoleculeState], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[MoleculeState], None]] = None

        # Early-stop support.
        self._stop_requested = False

    def request_stop(self) -> None:
        """Thread-safe-ish early-stop signal, checked between iterations."""
        self._stop_requested = True

    def _status(self, msg: str) -> None:
        if self.on_status is not None:
            try:
                self.on_status(msg)
            except Exception as e:
                logger.warning(f"on_status callback failed: {e}")

    def calculate_mape(self, molecule: MoleculeState) -> float:
        """MAPE (%) across the four target properties. Lower is better."""
        target_dict = self.target.to_dict()
        props = molecule.properties

        errors = []
        for key in ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid']:
            if key in target_dict and key in props:
                target_val = abs(target_dict[key])
                if target_val > 0:
                    errors.append(abs(props[key] - target_dict[key]) / target_val * 100)

        return sum(errors) / len(errors) if errors else 100.0

    def run(self, seed_molecule: MoleculeState) -> MoleculeState:
        """Run beam search and return the best molecule found."""
        current_beam = [seed_molecule]
        self.best_ever = seed_molecule
        prev_best_mape = self.calculate_mape(seed_molecule)

        print()
        print(f"   🌱 Seed Molecule: {seed_molecule.smiles[:50]}{'...' if len(seed_molecule.smiles) > 50 else ''}")
        print(f"      Initial MAPE: {prev_best_mape:.2f}%")
        print()

        logger.info(f"Starting beam search with seed: {seed_molecule.smiles}")
        if self.on_seed is not None:
            try:
                self.on_seed(seed_molecule)
            except Exception as e:
                logger.warning(f"on_seed callback failed: {e}")

        for iteration in range(self.beam_config.max_iterations):
            if self._stop_requested:
                self._status("Stop requested; exiting beam search.")
                break

            print()
            header = f"ITERATION {iteration + 1}/{self.beam_config.max_iterations}"
            print(f"┌{'─' * 58}┐")
            print(f"│  📍 {header}" + " " * max(0, 58 - 5 - len(header)) + "│")
            print(f"└{'─' * 58}┘")
            self._status(f"Iteration {iteration + 1}/{self.beam_config.max_iterations}")

            all_candidates: List[MoleculeState] = []

            for idx, parent_mol in enumerate(current_beam):
                print(f"\n   🔬 Parent {idx + 1}/{len(current_beam)}: "
                      f"{parent_mol.smiles[:45]}{'...' if len(parent_mol.smiles) > 45 else ''}")

                agent = ChemistAgent(
                    parent_mol,
                    self.target,
                    self.config,
                    predictor=self.predictor,
                    literature_retriever=self.literature_retriever,
                )
                new_candidates = agent.generate_variations()
                all_candidates.extend(new_candidates)
                print(f"      ✓ {len(new_candidates)} candidates generated")

            print(f"\n   📈 Iteration stats: total={len(all_candidates)}", end='')
            feasible = [m for m in all_candidates if m.is_feasible]
            print(f", feasible={len(feasible)}", end='')

            if not feasible:
                print("\n   ⚠️  No feasible candidates. Stopping.")
                break

            unique = self._remove_duplicates(feasible)
            print(f", unique={len(unique)}")

            ranked = sorted(unique, key=self.calculate_mape)
            next_beam = ranked[:self.beam_config.top_k]

            self.log_iteration(iteration + 1, next_beam)

            best_mape = self.calculate_mape(next_beam[0])
            best_ever_mape = self.calculate_mape(self.best_ever)
            if best_mape < best_ever_mape:
                self.best_ever = next_beam[0]
                print(f"\n   🌟 NEW BEST: MAPE {best_mape:.2f}%  "
                      f"{self.best_ever.smiles[:45]}{'...' if len(self.best_ever.smiles) > 45 else ''}")
                if self.on_best is not None:
                    try:
                        self.on_best(self.best_ever)
                    except Exception as e:
                        logger.warning(f"on_best callback failed: {e}")

            # --- Stop conditions ---------------------------------------------
            improvement = prev_best_mape - best_mape
            print(f"\n   📉 MAPE improvement: {improvement:+.3f}% "
                  f"(prev {prev_best_mape:.2f}% → now {best_mape:.2f}%)")

            # Absolute target reached? Use the globally-best MAPE so we don't
            # miss a target that was hit in an earlier iteration.
            target = float(getattr(self.beam_config, 'mape_target', 0.0) or 0.0)
            best_ever_mape = self.calculate_mape(self.best_ever)
            if target > 0 and best_ever_mape <= target:
                print(f"\n   ✅ MAPE target reached: {best_ever_mape:.2f}% ≤ {target:.2f}%")
                self._status(f"Target reached: MAPE {best_ever_mape:.2f}% ≤ {target:.2f}%")
                self._fire_iteration(iteration + 1, all_candidates, next_beam)
                current_beam = next_beam
                self.history.append(current_beam)
                break

            # Patience-based plateau check — only applies when no explicit
            # target was set. If the user set mape_target, honor it: keep
            # searching until target or max_iterations.
            if target <= 0:
                if iteration > 0 and improvement < self.beam_config.convergence_threshold:
                    self._stall_count = getattr(self, '_stall_count', 0) + 1
                else:
                    self._stall_count = 0
                patience = max(1, int(getattr(self.beam_config, 'patience', 2)))
                if self._stall_count >= patience:
                    print(f"\n   ✅ Converged: no meaningful MAPE improvement for "
                          f"{patience} iterations (Δ < {self.beam_config.convergence_threshold}%).")
                    self._fire_iteration(iteration + 1, all_candidates, next_beam)
                    current_beam = next_beam
                    self.history.append(current_beam)
                    break

            prev_best_mape = best_mape
            current_beam = next_beam
            self.history.append(current_beam)
            self._fire_iteration(iteration + 1, all_candidates, next_beam)

        print()
        print(f"┌{'─' * 58}┐")
        print(f"│  ✅ BEAM SEARCH COMPLETE" + " " * 33 + "│")
        print(f"└{'─' * 58}┘")
        print()

        logger.info(f"Beam search complete. Best: {self.best_ever.smiles}")
        if self.on_complete is not None:
            try:
                self.on_complete(self.best_ever)
            except Exception as e:
                logger.warning(f"on_complete callback failed: {e}")

        return self.best_ever

    def _fire_iteration(self, iteration: int,
                        all_candidates: List[MoleculeState],
                        beam: List[MoleculeState]) -> None:
        if self.on_iteration is None:
            return
        try:
            self.on_iteration(iteration, all_candidates, beam)
        except Exception as e:
            logger.warning(f"on_iteration callback failed: {e}")

    def _remove_duplicates(self, molecules: List[MoleculeState]) -> List[MoleculeState]:
        seen = set()
        unique = []
        for mol in molecules:
            if mol.smiles not in seen:
                seen.add(mol.smiles)
                unique.append(mol)
        return unique

    def log_iteration(self, iteration: int, beam: List[MoleculeState]) -> None:
        print(f"\n   🏅 Top {min(3, len(beam))} this iteration:")
        for i, mol in enumerate(beam[:3]):
            mape = self.calculate_mape(mol)
            feasibility_pct = (1 - mol.feasibility) * 100
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            print(f"      {medal} #{i+1}: MAPE {mape:.1f}%  feas {feasibility_pct:.0f}%  "
                  f"{mol.smiles[:40]}{'...' if len(mol.smiles) > 40 else ''}")
        logger.debug(f"Iteration {iteration}: beam size = {len(beam)}")
