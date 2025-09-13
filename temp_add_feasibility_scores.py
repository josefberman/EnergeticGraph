import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors as RDDesc
    from rdkit.Chem import rdMolDescriptors as RDMD
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
except Exception as e:
    Chem = None  # type: ignore


def smiles_to_mol_3d(smiles: str, n_confs: int = 8):
    if Chem is None:
        return None, []
    m0 = Chem.MolFromSmiles(smiles)
    if m0 is None:
        return None, []
    mol = Chem.AddHs(m0)
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.5
    try:
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
        # initial geometry optimization
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s")
    except Exception:
        cids = []
    return mol, cids


def mmff_strain_kcal(mol, cids):
    if Chem is None or mol is None or not cids:
        return {}
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s")
        energies = {cid: e[1] for cid, e in zip(cids, res)}
        if not energies:
            return {}
        e0 = min(energies.values())
        return {cid: energies[cid] - e0 for cid in cids}
    except Exception:
        return {}


def ersatz_sa_score(mol) -> float:
    if Chem is None or mol is None:
        return 5.0
    try:
        rings = RDMD.CalcNumRings(mol)
        spiro = RDMD.CalcNumSpiroAtoms(mol)
        bridged = RDMD.CalcNumBridgeheadAtoms(mol)
        size = mol.GetNumAtoms()
        heteros = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        return min(10.0, 1.5 + 0.03 * size + 0.6 * rings + 0.8 * spiro + 0.8 * bridged + 0.2 * heteros)
    except Exception:
        return 5.0


def count_alerts(mol) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        fc = FilterCatalog(params)
        return sum(1 for _ in fc.GetMatches(mol))
    except Exception:
        return 0


def compute_scores_for_smiles(smiles: str, n_confs: int = 8):
    # Build molecule and conformers
    mol, cids = smiles_to_mol_3d(smiles, n_confs=n_confs)
    if mol is None:
        return np.nan, np.nan

    # Basic features
    sa = ersatz_sa_score(mol)
    alerts = count_alerts(mol)
    try:
        ring_info = mol.GetRingInfo()
        small_rings = sum(1 for r in ring_info.AtomRings() if len(r) in (3, 4))
    except Exception:
        small_rings = 0
    try:
        rotb = RDDesc.NumRotatableBonds(mol)
    except Exception:
        rotb = 0

    # MMFF strain across conformers
    strain_map = mmff_strain_kcal(mol, cids)
    strains = [strain_map[c] for c in cids] if cids and strain_map else [0.0]
    strain_min = float(min(strains)) if strains else 0.0
    strain_p95 = float(np.percentile(strains, 95)) if len(strains) > 1 else strain_min

    # Composite feasibility score (0..1), matching agent's heuristic with safe fallbacks
    # Skip BDE and quantum terms; align with agent's fallback (min_bde -> 0, eta -> ~0)
    min_bde = 0.0
    eta = 0.0

    s = 1.0
    s -= 0.06 * max(0.0, sa - 3.0)
    s -= 0.15 * min(alerts, 2)
    s -= 0.02 * max(0.0, strain_p95 - 2.0)
    s -= 0.04 * max(0.0, (5.0 - min_bde) / 5.0)
    s -= 0.02 * float(small_rings)
    s -= 0.01 * max(0.0, float(rotb) - 8.0)
    s += 0.005 * min(20.0, float(eta)) / 10.0
    s = max(0.0, min(1.0, s))

    # MMFF-only feasibility from strain
    strain_only_feas = max(0.0, min(1.0, 1.0 - 0.02 * max(0.0, strain_p95 - 2.0)))

    return round(s, 3), round(strain_only_feas, 3)


def process_csv(path: Path, n_confs: int = 8) -> None:
    df = pd.read_csv(path)
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column")

    composites = []
    mmff_scores = []
    for smi in df['SMILES'].astype(str).tolist():
        try:
            c, m = compute_scores_for_smiles(smi, n_confs=n_confs)
        except Exception:
            c, m = np.nan, np.nan
        composites.append(c)
        mmff_scores.append(m)

    df['composite_score_0_1'] = composites
    df['mmff_strain_score_0_1'] = mmff_scores

    # Backup before overwrite
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    df.to_csv(backup, index=False)
    # Overwrite original
    df.to_csv(path, index=False)
    print(f"Wrote new columns to: {path}")
    print(f"Backup saved at: {backup}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add feasibility scores to a molecules CSV")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="sample_start_molecules.csv",
        help="Path to input CSV with a 'SMILES' column",
    )
    parser.add_argument(
        "--confs",
        type=int,
        default=8,
        help="Number of conformers for MMFF strain calculation",
    )
    args = parser.parse_args(argv)

    path = Path(args.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    process_csv(path, n_confs=args.confs)


if __name__ == "__main__":
    main()


