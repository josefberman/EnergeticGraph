"""
Property prediction module using XGBoost models.
"""

import os
import joblib
import numpy as np
from typing import Dict, Optional
import logging

from descriptors import create_descriptor

logger = logging.getLogger(__name__)


class PropertyPredictor:
    """Predicts molecular properties using trained XGBoost models."""
    
    def __init__(self, models_directory: str):
        """
        Initialize predictor and load models.
        
        Args:
            models_directory: Path to directory containing .joblib model files
        """
        self.models_directory = models_directory
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all XGBoost models from directory."""
        property_names = [
            'hf_solid',
            'det_velocity',
            'det_pressure',
            'density'
        ]
        
        for prop_name in property_names:
            model_path = os.path.join(self.models_directory, f"{prop_name}.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[prop_name] = joblib.load(model_path)
                    logger.info(f"Loaded model for {prop_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {prop_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if not self.models:
            raise ValueError(f"No models loaded from {self.models_directory}")
    
    def predict_properties(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Predict all properties for a given SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Dictionary with property predictions, or None if descriptor generation fails
        """
        # Generate descriptors
        desc = create_descriptor(smiles)
        if desc is None:
            logger.warning(f"Failed to create descriptor for {smiles}")
            return None
        
        # Reshape for prediction
        X = np.array([desc])
        
        predictions = {}
        
        # Map internal names to user-facing names
        property_mapping = {
            'hf_solid': 'Hf solid',
            'det_velocity': 'Det Velocity',
            'det_pressure': 'Det Pressure',
            'density': 'Density'
        }
        
        for internal_name, display_name in property_mapping.items():
            if internal_name in self.models:
                try:
                    pred = self.models[internal_name].predict(X)[0]
                    predictions[display_name] = float(pred)
                except Exception as e:
                    logger.error(f"Error predicting {display_name}: {e}")
                    predictions[display_name] = None
            else:
                predictions[display_name] = None
        
        return predictions


def predict_properties(smiles: str, models_directory: str = "./beam_search_system/models") -> Optional[Dict[str, float]]:
    """
    Convenience function to predict properties without creating predictor instance.
    
    Args:
        smiles: SMILES string
        models_directory: Path to models
        
    Returns:
        Dictionary of predictions
    """
    predictor = PropertyPredictor(models_directory)
    return predictor.predict_properties(smiles)
