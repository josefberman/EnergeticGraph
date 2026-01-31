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


def calculate_mape(predicted_props: Dict[str, float], 
                   target_props: Dict[str, float],
                   property_weights: Dict[str, float] = None) -> float:
    """
    Calculate weighted Mean Absolute Percentage Error between predicted and target properties.
    
    MAPE = (1/n) * Σ(|predicted - target| / |target|) * 100
    
    Args:
        predicted_props: Predicted property values
        target_props: Target property values
        property_weights: Weight for each property (should sum to 1.0)
        
    Returns:
        Weighted MAPE as percentage (lower is better)
    """
    if property_weights is None:
        property_weights = {
            'Density': 0.25,
            'Det Velocity': 0.25,
            'Det Pressure': 0.25,
            'Hf solid': 0.25
        }
    
    mape = 0.0
    total_weight = 0.0
    
    for prop_name in target_props.keys():
        if prop_name in predicted_props and predicted_props[prop_name] is not None:
            pred = predicted_props[prop_name]
            target = target_props[prop_name]
            weight = property_weights.get(prop_name, 0.25)
            
            # Calculate percentage error relative to target
            if abs(target) > 1e-10:  # Avoid division by zero
                percentage_error = abs(pred - target) / abs(target) * 100
            else:
                # If target is near zero, use absolute error
                percentage_error = abs(pred - target) * 100
            
            mape += weight * percentage_error
            total_weight += weight
        else:
            logger.warning(f"Property {prop_name} missing in predictions")
    
    # Normalize by total weight
    if total_weight > 0:
        mape /= total_weight
    
    return mape


def calculate_total_score(predicted_props: Dict[str, float],
                          target_props: Dict[str, float],
                          normalized_sascore: float,
                          mape_weight: float = 0.7,
                          sascore_weight: float = 0.3,
                          property_weights: Dict[str, float] = None) -> float:
    """
    Calculate total score combining MAPE and normalized SAScore.
    
    Score = mape_weight * (MAPE/100) + sascore_weight * normalized_sascore
    
    Both components are 0-1 where lower is better, so the total score
    is minimized for better candidates.
    
    Args:
        predicted_props: Predicted properties
        target_props: Target properties
        normalized_sascore: Normalized SAScore (0-1, 0 = most feasible, 1 = least feasible)
        mape_weight: Weight for property accuracy (default 0.7)
        sascore_weight: Weight for synthetic accessibility (default 0.3)
        property_weights: Weights for each property in MAPE calculation
        
    Returns:
        Total score (lower is better)
    """
    # Default weights
    if property_weights is None:
        property_weights = {
            'Density': 0.25,
            'Det Velocity': 0.25,
            'Det Pressure': 0.25,
            'Hf solid': 0.25
        }
    
    # Calculate MAPE (as percentage)
    mape = calculate_mape(predicted_props, target_props, property_weights)
    
    # Convert MAPE percentage to 0-1 scale for scoring (cap at 100%)
    mape_normalized = min(mape / 100.0, 1.0)
    
    # Combined score: both terms are 0-1 where lower is better
    total_score = mape_weight * mape_normalized + sascore_weight * normalized_sascore
    
    return total_score
