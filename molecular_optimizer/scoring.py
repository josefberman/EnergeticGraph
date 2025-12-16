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
        """Calculate combined score that minimizes MAPE and maximizes feasibility.
        
        Multi-objective optimization:
        - Minimize property error (MAPE/MSE) - 70% weight
        - Maximize feasibility (synthetic accessibility) - 30% weight
        
        Args:
            property_error: Property error (MAPE or MSE)
            feasibility: Feasibility report
        
        Returns:
            Combined score (lower is better) - weighted sum for beam search minimization
        """
        if feasibility is None:
            return 999.0  # Very bad score if no feasibility
        
        # Get feasibility score (0-1, higher is better)
        feas_score = feasibility.composite_score_0_1
        
        # Normalize property error to 0-1 range
        norm_prop_error = min(property_error, 1.0)
        
        # Invert feasibility for minimization (higher feasibility → lower score)
        inverted_feas = 1.0 - feas_score
        
        # Weighted combination: 70% property error, 30% feasibility
        combined = 0.7 * norm_prop_error + 0.3 * inverted_feas
        
        return float(combined)
    
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
