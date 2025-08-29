"""
Script to retrain the neural network model and scaler used for prediction.
Trains the NN on Coulomb matrix eigenvalue descriptors (length 64) and
produces multi-output predictions for the five energetic properties.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

import pandas as pd
import os
from prediction import train_data

def retrain_models():
    """Retrain the neural network model and scaler used for predictions."""
    print("Retraining XGBoost-based models for energetic property prediction...")
    
    # Check if data file exists
    data_file = "extracted_chemical_data.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please ensure the data file is available.")
        return False
    
    try:
        # Load data
        print("Loading data from extracted_chemical_data.csv ...")
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples")

        if not os.path.exists('./trained_models/'):
            os.makedirs('./trained_models/')
        if not os.path.exists('./trained_models_plots/'):
            os.makedirs('./trained_models_plots/')

        # Retrain XGBoost models (per property) and save scalers/models
        train_data(df)
        
        print("\n✅ XGBoost model retraining completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model retraining: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("MODEL RETRAINING SCRIPT")
    print("=" * 60)
    
    success = retrain_models()
    
    if success:
        print("\n🎉 Models have been retrained with the current scikit-learn version!")
        print("You can now run the molecular optimization system without version warnings.")
    else:
        print("\n⚠️  Model retraining failed. Check the error messages above.")

if __name__ == "__main__":
    main() 