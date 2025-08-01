#!/usr/bin/env python3
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

def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def retrain_models():
    """Retrain all models with current scikit-learn version"""
    print("Retraining models with current scikit-learn version...")
    
    # Check if data file exists
    data_file = "clean_data_imputed.xlsx"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please ensure the data file is available.")
        return False
    
    try:
        # Load data
        print("Loading data...")
        df = pd.read_excel(data_file)
        print(f"Loaded {len(df)} samples")
        
        # Create descriptors
        print("Creating descriptors...")
        from prediction import create_descriptor
        
        # Clean SMILES data and handle errors
        valid_smiles = []
        valid_descriptors = []
        
        for idx, smiles in enumerate(df['SMILES']):
            try:
                if pd.isna(smiles) or not isinstance(smiles, str):
                    continue
                descriptor = create_descriptor(smiles)
                if descriptor is not None and len(descriptor) > 0:
                    valid_smiles.append(smiles)
                    valid_descriptors.append(descriptor)
            except Exception as e:
                print(f"  Skipping invalid SMILES at index {idx}: {e}")
                continue
        
        print(f"  Created {len(valid_descriptors)} valid descriptors from {len(df)} samples")
        
        if len(valid_descriptors) == 0:
            print("❌ No valid descriptors created. Check your SMILES data.")
            return False
        
        X = np.array(valid_descriptors)
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        print(f"Descriptor shape: {X.shape}")
        
        # Create output directory
        os.makedirs('./trained_models', exist_ok=True)
        os.makedirs('./trained_models_plots', exist_ok=True)
        
        # Create a new dataframe with valid data
        print("Creating valid dataframe...")
        
        # Get the original dataframe indices for valid SMILES
        valid_indices = []
        for smiles in valid_smiles:
            matches = df[df['SMILES'] == smiles].index
            if len(matches) > 0:
                valid_indices.append(matches[0])
        
        # Create valid_df by filtering the original dataframe
        valid_df = df.loc[valid_indices].copy()
        valid_df.reset_index(drop=True, inplace=True)
        
        print(f"Valid dataframe shape: {valid_df.shape}")
        print(f"Valid dataframe columns: {list(valid_df.columns)}")
        
        # Train models for each property
        property_columns = ['Density', 'Detonation velocity', 'Explosion capacity', 
                           'Explosion pressure', 'Explosion heat', 'Solid phase formation enthalpy']
        
        for col in property_columns:
            print(f"\nChecking column: {col}")
            print(f"Available columns: {list(valid_df.columns)}")
            
            if col in valid_df.columns:
                print(f"Training model for {col}...")
                
                # Check data availability
                y = valid_df[col].dropna()
                print(f"  Available {col} data: {len(y)} samples")
                
                if len(y) == 0:
                    print(f"  No valid data for {col}, skipping...")
                    continue
                
                # Get corresponding X data
                valid_mask = ~valid_df[col].isna()
                X_subset = X[valid_mask]
                
                print(f"  X_subset shape: {X_subset.shape}")
                print(f"  y shape: {y.shape}")
                
                if len(X_subset) != len(y):
                    print(f"  Warning: X and y lengths don't match ({len(X_subset)} vs {len(y)})")
                    # Align the data
                    min_len = min(len(X_subset), len(y))
                    X_subset = X_subset[:min_len]
                    y = y[:min_len]
                
                # Split data
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_subset, y, test_size=0.2, random_state=42
                    )
                    
                    print(f"  Train set: {len(X_train)} samples")
                    print(f"  Test set: {len(X_test)} samples")
                    
                    # Scale features
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                except Exception as e:
                    print(f"  Error in data splitting/scaling: {e}")
                    continue
                
                # Save scaler
                with open('./trained_models/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                
                # Train model
                y_train_mean = np.mean(y_train)
                
                # Try different models
                models = [
                    ("KernelRidge", KernelRidge(), {
                        "alpha": np.logspace(-10, 2, 20),
                        "gamma": np.logspace(-10, -1, 20),
                        "kernel": ['rbf']
                    }),
                    ("RandomForest", RandomForestRegressor(), {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [3, 5, 10, None]
                    }),
                    ("SVR", SVR(), {
                        "kernel": ["rbf"],
                        "C": np.logspace(-2, 2, 5),
                        "gamma": np.logspace(-10, -1, 5)
                    })
                ]
                
                best_score = -np.inf
                best_model = None
                best_model_name = None
                
                for model_name, model, param_grid in models:
                    try:
                        print(f"  Trying {model_name}...")
                        grd_srch = GridSearchCV(
                            model, param_grid, cv=3, scoring='neg_mean_squared_error', 
                            verbose=0, n_jobs=-1
                        )
                        grd_srch.fit(X_train_scaled, y_train)
                        
                        if grd_srch.best_score_ > best_score:
                            best_score = grd_srch.best_score_
                            best_model = grd_srch.best_estimator_
                            best_model_name = model_name
                            
                    except Exception as e:
                        print(f"    {model_name} failed: {e}")
                        continue
                
                if best_model is not None:
                    # Save best model
                    with open(f'./trained_models/{col}.pkl', 'wb') as f:
                        pickle.dump(best_model, f)
                    
                    # Evaluate model
                    y_pred = best_model.predict(X_test_scaled)
                    
                    print(f"  Best model: {best_model_name}")
                    print(f"  Train R² score: {r2_score(y_train, best_model.predict(X_train_scaled)):.4f}")
                    print(f"  Test R² score: {r2_score(y_test, y_pred):.4f}")
                    print(f"  Test RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
                    print(f"  Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
                    
                    # Create plots
                    plt.figure(figsize=(8, 4))
                    
                    # Prediction plot
                    plt.subplot(1, 2, 1)
                    plt.scatter(y_test, y_pred, s=20, alpha=0.6, c='#e63946')
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
                    plt.xlabel('Actual values')
                    plt.ylabel('Predicted values')
                    plt.title(f'{col} - Predictions')
                    
                    # Residuals plot
                    plt.subplot(1, 2, 2)
                    residuals = y_pred - y_test
                    plt.scatter(y_test, residuals, s=20, alpha=0.6, c='#457b9d')
                    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
                    plt.xlabel('Actual values')
                    plt.ylabel('Residuals')
                    plt.title(f'{col} - Residuals')
                    
                    plt.tight_layout()
                    plt.savefig(f'./trained_models_plots/{col}.jpg', dpi=200, bbox_inches='tight')
                    plt.close()
                    
                else:
                    print(f"  No valid model found for {col}")
        
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