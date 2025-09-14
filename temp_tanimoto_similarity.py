import argparse
import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except Exception:
    Chem = None  # type: ignore


def canonicalize_smiles(smiles: str) -> Tuple[str, Optional['Chem.Mol']]:
    if Chem is None:
        return smiles, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        can = Chem.MolToSmiles(mol, canonical=True)
        return can, mol
    except Exception:
        return smiles, None


def morgan_fp_2048(mol: Optional['Chem.Mol']):
    if Chem is None or mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return float('nan')
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return float('nan')


def auto_pick_smiles_columns(df: pd.DataFrame, col1: Optional[str], col2: Optional[str]) -> Tuple[str, str]:
    if col1 and col2:
        return col1, col2
    # Fallback: take first two columns
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("CSV must contain at least two columns for SMILES")
    return cols[0], cols[1]


def process_csv(path: Path, col1: Optional[str], col2: Optional[str], out_col: str = 'tanimoto_morgan2048') -> None:
    df = pd.read_csv(path)
    c1, c2 = auto_pick_smiles_columns(df, col1, col2)

    sims = []
    can1_vals = []
    can2_vals = []

    for s1, s2 in zip(df[c1].astype(str), df[c2].astype(str)):
        can1, m1 = canonicalize_smiles(s1.strip())
        can2, m2 = canonicalize_smiles(s2.strip())
        fp1 = morgan_fp_2048(m1)
        fp2 = morgan_fp_2048(m2)
        sim = tanimoto(fp1, fp2)
        can1_vals.append(can1)
        can2_vals.append(can2)
        sims.append(sim)

    # Overwrite the two SMILES columns with canonicalized SMILES
    df[c1] = can1_vals
    df[c2] = can2_vals
    df[out_col] = sims

    # Backup and overwrite
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    df.to_csv(backup, index=False)
    df.to_csv(path, index=False)
    print(f"Wrote Tanimoto similarities to '{out_col}' in: {path}")
    print(f"Backup saved at: {backup}")


def main():
    parser = argparse.ArgumentParser(description='Compute Tanimoto similarity (Morgan 2048-bit) for two SMILES columns in a CSV')
    parser.add_argument('csv_path', help='Path to input CSV')
    parser.add_argument('--col1', help='Name of first SMILES column (default: first column)')
    parser.add_argument('--col2', help='Name of second SMILES column (default: second column)')
    parser.add_argument('--out-col', default='tanimoto_morgan2048', help='Output column name for similarity')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    process_csv(csv_path, args.col1, args.col2, out_col=args.out_col)


if __name__ == '__main__':
    main()


