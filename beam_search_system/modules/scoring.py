"""
Scoring functions for molecular design.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def normalize_property(value: float, prop_range: tuple) -> float:
    """
    Normalize a property value to 0-1 range.
    
    Args:
        value: Property value
        prop_range: (min, max) tuple for normalization
        
    Returns:
        Normalized value
    """
    min_val, max_val = prop_range
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def calculate_mae(predicted_props: Dict[str, float], 
                  target_props: Dict[str, float],
                  property_weights: Dict[str, float],
                  property_ranges: Dict[str, tuple]) -> float:
    """
    Calculate weighted Mean Absolute Error between predicted and target properties.
    
    Args:
        predicted_props: Predicted property values
        target_props: Target property values
        property_weights: Weight for each property (should sum to 1.0)
        property_ranges: Normalization ranges for each property
        
    Returns:
        Weighted MAE (lower is better)
    """
    mae = 0.0
    total_weight = 0.0
    
    for prop_name in target_props.keys():
        if prop_name in predicted_props and predicted_props[prop_name] is not None:
            # Get values
            pred = predicted_props[prop_name]
            target = target_props[prop_name]
            weight = property_weights.get(prop_name, 0.25)
            prop_range = property_ranges.get(prop_name, (0, 1))
            
            # Normalize both values
            pred_norm = normalize_property(pred, prop_range)
            target_norm = normalize_property(target, prop_range)
            
            # Calculate absolute error
            error = abs(pred_norm - target_norm)
            mae += weight * error
            total_weight += weight
        else:
            logger.warning(f"Property {prop_name} missing in predictions")
    
    # Normalize by total weight
    if total_weight > 0:
        mae /= total_weight
    
    return mae


def calculate_total_score(predicted_props: Dict[str, float],
                          target_props: Dict[str, float],
                          feasibility: float,
                          mae_weight: float = 0.7,
                          feasibility_weight: float = 0.3,
                          property_weights: Dict[str, float] = None,
                          property_ranges: Dict[str, tuple] = None) -> float:
    """
    Calculate total score combining MAE and feasibility.
    
    Score = mae_weight * MAE + feasibility_weight * (1 - feasibility)
    
    Args:
        predicted_props: Predicted properties
        target_props: Target properties
        feasibility: Feasibility score (0-1, higher is better)
        mae_weight: Weight for property accuracy (default 0.7)
        feasibility_weight: Weight for feasibility (default 0.3)
        property_weights: Weights for each property in MAE calculation
        property_ranges: Normalization ranges for properties
        
    Returns:
        Total score (lower is better)
    """
    # Default weights and ranges
    if property_weights is None:
        property_weights = {
            'Density': 0.25,
            'Det Velocity': 0.25,
            'Det Pressure': 0.25,
            'Hf solid': 0.25
        }
    
    if property_ranges is None:
        property_ranges = {
            'Density': (1.0, 2.5),
            'Det Velocity': (6000.0, 10000.0),
            'Det Pressure': (10.0, 50.0),
            'Hf solid': (-500.0, 500.0)
        }
    
    # Calculate MAE
    mae = calculate_mae(predicted_props, target_props, property_weights, property_ranges)
    
    # Calculate feasibility penalty (1 - feasibility)
    # Higher feasibility = lower penalty
    feasibility_penalty = 1.0 - feasibility
    
    # Combined score
    total_score = mae_weight * mae + feasibility_weight * feasibility_penalty
    
    return total_score
