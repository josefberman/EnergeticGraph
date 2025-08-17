"""
Script to retrain the models with the current scikit-learn version
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prediction import train_data

def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def retrain_models():
    """Retrain all models with current scikit-learn version"""
    print("Retraining models with current scikit-learn version...")
    
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
        if len(os.listdir('./trained_models/')) == 0:
            train_data(df)
        
        print("\n✅ Model retraining completed successfully!")
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