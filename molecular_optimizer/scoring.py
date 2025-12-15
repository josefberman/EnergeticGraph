"""Fitness and scoring functions for optimization.

This module handles property error calculations (MAPE/MSE)
and combined scoring logic.
"""
from typing import Dict, Optional

from .state import FeasibilityReport


class ScoringCalculator:
    """Calculate fitness scores for molecular candidates."""
    
    def __init__(self, error_metric: str = 'mape'):
        """Initialize scoring calculator.
        
        Args:
            error_metric: 'mape' for Mean Absolute Percentage Error or 'mse' for Mean Squared Error
        """
        self.error_metric = error_metric.lower()
    
    def calculate_property_error(
        self,
        properties: Dict[str, float],
        target_properties: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted property error (MAPE or MSE).
        
        Args:
            properties: Current molecule properties
            target_properties: Target property values
            weights: Weight for each property
        
        Returns:
            Weighted error (lower is better)
        """
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
    
    def calculate_combined_score(
        self,
        property_error: float,
        feasibility: Optional[FeasibilityReport]
    ) -> float:
        """Calculate combined score for candidate ranking.
        
        Note: Feasibility is enforced via threshold filter elsewhere,
        so we just return property error for ranking.
        
        Args:
            property_error: Property error (MAPE or MSE)
            feasibility: Feasibility report (not used in scoring, just for filtering)
        
        Returns:
            Combined score (lower is better) - currently just property error
        """
        return float(property_error)
    
    def feasibility_to_dict(
        self,
        feas: Optional[FeasibilityReport]
    ) -> Optional[Dict]:
        """Convert feasibility report to dictionary for reporting.
        
        Args:
            feas: Feasibility report
        
        Returns:
            Dictionary with feasibility fields, or None
        """
        if feas is None:
            return None
        return feas.__dict__
