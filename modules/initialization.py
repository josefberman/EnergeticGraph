"""
Initialization module - finds best seed molecule from dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from data_structures import MoleculeState, PropertyTarget

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load molecular dataset with properties.
    
    Args:
        dataset_path: Path to dataset file (.xlsx or .csv)
        
    Returns:
        DataFrame with SMILES and properties
    """
    try:
        if dataset_path.endswith('.xlsx'):
            df = pd.read_excel(dataset_path, sheet_name='data')
        else:
            df = pd.read_csv(dataset_path)
        
        logger.info(f"Loaded dataset with {len(df)} molecules from {dataset_path}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def calculate_euclidean_distance(props1: Dict[str, float], 
                                 props2: Dict[str, float],
                                 property_ranges: Dict[str, tuple]) -> float:
    """
    Calculate Euclidean distance between two property vectors.
    
    Args:
        props1: First property dict
        props2: Second property dict
        property_ranges: Normalization ranges
        
    Returns:
        Euclidean distance (normalized)
    """
    squared_diff_sum = 0.0
    count = 0
    
    for prop_name in props1.keys():
        if prop_name in props2:
            val1 = props1[prop_name]
            val2 = props2[prop_name]
            
            # Normalize
            if prop_name in property_ranges:
                min_val, max_val = property_ranges[prop_name]
                if max_val != min_val:
                    val1 = (val1 - min_val) / (max_val - min_val)
                    val2 = (val2 - min_val) / (max_val - min_val)
            
            squared_diff_sum += (val1 - val2) ** 2
            count += 1
    
    if count == 0:
        return float('inf')
    
    return np.sqrt(squared_diff_sum / count)


def find_closest_match(dataset: pd.DataFrame,
                       target_properties: PropertyTarget,
                       property_ranges: Dict[str, tuple]) -> MoleculeState:
    """
    Find the closest molecule in dataset to target properties.
    
    Args:
        dataset: DataFrame with SMILES and properties
        target_properties: Target property values
        property_ranges: Normalization ranges for distance calculation
        
    Returns:
        MoleculeState of best seed molecule
    """
    target_dict = target_properties.to_dict()
    
    # Map dataset columns to property names
    # CSV columns: 'density', 'det_velocity', 'det_pressure', 'hf_solid', 'SMILES'
    column_mapping = {
        'hf_solid': 'Hf solid',
        'det_velocity': 'Det Velocity',
        'det_pressure': 'Det Pressure',
        'density': 'Density'
    }
    
    best_smiles = None
    best_distance = float('inf')
    best_props = None
    
    for idx, row in dataset.iterrows():
        # Extract properties
        props = {}
        valid = True
        for col, prop_name in column_mapping.items():
            if col in row:
                val = row[col]
                # Skip if NaN or empty
                if pd.isna(val) or val == '':
                    valid = False
                    break
                try:
                    props[prop_name] = float(val)
                except (ValueError, TypeError):
                    valid = False
                    break
        
        if not valid or len(props) != len(column_mapping):
            continue
        
        # Calculate distance
        distance = calculate_euclidean_distance(target_dict, props, property_ranges)
        
        if distance < best_distance:
            best_distance = distance
            best_smiles = row['SMILES']
            best_props = props
    
    if best_smiles is None:
        raise ValueError("No valid seed molecule found in dataset")
    
    logger.info(f"Found best seed: {best_smiles} with distance {best_distance:.4f}")
    logger.info(f"Seed properties: {best_props}")
    
    # Create MoleculeState
    seed = MoleculeState(
        smiles=best_smiles,
        properties=best_props,
        score=best_distance,  # Initial score is distance to target
        feasibility=1.0,  # Assume dataset molecules are feasible
        is_feasible=True,
        provenance="seed_from_dataset",
        generation=0
    )
    
    return seed
