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


def calculate_mape_distance(props1: Dict[str, float], 
                            props2: Dict[str, float]) -> float:
    """
    Calculate MAPE-based distance between two property vectors.
    Uses Mean Absolute Percentage Error - no normalization needed.
    
    Args:
        props1: First property dict (target)
        props2: Second property dict (candidate)
        
    Returns:
        MAPE distance (lower is better)
    """
    total_error = 0.0
    count = 0
    
    for prop_name in props1.keys():
        if prop_name in props2:
            target = props1[prop_name]
            pred = props2[prop_name]
            
            # Calculate percentage error relative to target
            if abs(target) > 1e-10:
                percentage_error = abs(target - pred) / abs(target) * 100
            else:
                percentage_error = abs(target - pred) * 100
            
            total_error += percentage_error
            count += 1
    
    if count == 0:
        return float('inf')
    
    return total_error / count


def find_closest_match(dataset: pd.DataFrame,
                       target_properties: PropertyTarget) -> MoleculeState:
    """
    Find the closest molecule in dataset to target properties using MAPE.
    
    Args:
        dataset: DataFrame with SMILES and properties
        target_properties: Target property values
        
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
        
        # Calculate MAPE distance (no normalization needed)
        distance = calculate_mape_distance(target_dict, props)
        
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
        score=best_distance,
        feasibility=0.0,
        is_feasible=True,
        provenance="seed_from_dataset",
        generation=0,
        property_sources={k: 'dataset' for k in best_props.keys()},
    )
    
    return seed
