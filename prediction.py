import os
import joblib
import numpy as np
from urllib.parse import quote
from urllib.request import urlopen
from langchain_core.tools import tool
from descriptors import create_descriptor

@tool
def convert_name_to_smiles(name: str) -> str:
    """
    Converts a molecule's name to its SMILES representation.
    :param name: name of the molecule
    :return: SMILES representation of the molecule
    """
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not convert'

@tool
def predict_properties(smiles: str) -> dict:
    """
    Predicts the following properties of energetic materials: density, det_pressure, det_velocity, hf_solid.
    :param smiles: SMILES string representing the molecule to predict.
    :return: Dictionary with predicted values
    """
    try:
        # Generate descriptors
        desc = create_descriptor(smiles)
        
        if desc is None:
            print("Error: Could not generate descriptors.")
            return {}

        # Reshape for prediction (1 sample, n features)
        # Check if desc is already list or array
        X = np.array([desc])

        predictions = {}
        # Map target names (used in filenames) to output keys
        # Filenames are like 'hf_solid.joblib'
        # User wants keys: "density", "det_pressure", "det_velocity", "hf_solid"
        
        # Files in models/: density.joblib, det_pressure.joblib, det_velocity.joblib, hf_solid.joblib
        
        targets = [
            'density',
            'det_pressure',
            'det_velocity',
            'hf_solid'
        ]

        for target_key in targets:
            model_filename = f"{target_key}.joblib"
            model_path = os.path.join('models', model_filename)
            
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    pred = model.predict(X)[0]
                    predictions[target_key] = float(pred)
                except Exception as e:
                    print(f"Error predicting {target_key}: {e}")
                    predictions[target_key] = 0.0
            else:
                print(f"Warning: Model {model_filename} not found.")
                predictions[target_key] = 0.0
                
        return predictions

    except Exception as e:
        print(f"Error in predict_properties: {e}")
        return {
            'density': 0.0,
            'det_pressure': 0.0,
            'det_velocity': 0.0,
            'hf_solid': 0.0
        }

@tool
def predict_properties_nn(smiles: str) -> dict:
    """
    Predicts properties using Neural Network models.
    Currently falls back to standard predict_properties as NN models for new properties are not available.
    """
    return predict_properties(smiles)
