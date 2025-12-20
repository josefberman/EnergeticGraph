import argparse
from rdkit import Chem
from rdkit.DataStructs import DataStructs

def create_morgan_fingerprint(smiles: str):
    """
    Creates a Morgan fingerprint for a given SMILES string
    :param smiles: SMILES string
    :return: Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def calculate_similarity(smiles1: str, smiles2: str):
    """
    Calculates the Tanimoto similarity between two molecules
    :param smiles1: SMILES string of molecule 1
    :param smiles2: SMILES string of molecule 2
    :return: Tanimoto similarity
    """
    fp1 = create_morgan_fingerprint(smiles1)
    fp2 = create_morgan_fingerprint(smiles2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    """
    Calculates the Tanimoto similarity between two molecules
    """
    parser = argparse.ArgumentParser(description='Calculate Tanimoto similarity between two molecules')
    parser.add_argument('smiles1', type=str, help='SMILES string of molecule 1')
    parser.add_argument('smiles2', type=str, help='SMILES string of molecule 2')
    args = parser.parse_args()
    
    similarity = calculate_similarity(args.smiles1, args.smiles2)
    print(f'Tanimoto similarity: {similarity:.4f}')


if __name__ == '__main__':
    main()
