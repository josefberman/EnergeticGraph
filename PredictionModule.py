import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from descriptors import create_descriptor
from langchain_core.tools import tool


def import_data():
    """
    Imports the data from the excel file
    :return: DataFrame
    """
    df = pd.read_excel('new_data.xlsx', sheet_name='data')
    df = df[['SMILES', 'Hf solid', 'Det Velocity', 'Det Pressure', 'Density']]
    df.dropna(inplace=True)
    return df


def train_models(df: pd.DataFrame):
    """
    Trains the data using the XGBoost and SVM models
    :param df: DataFrame
    :return: None
    """
    df['desc'] = df['SMILES'].apply(create_descriptor)
    X = np.array(df['desc'].tolist())

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Targets configuration
    targets = {
        'Hf solid': 'xgb',
        'Det Velocity': 'xgb',
        'Det Pressure': 'xgb',
        'Density': 'xgb'
    }

    for target_name, model_type in targets.items():
        y = df[target_name].values
        print(f"\nTraining for target: {target_name} ({model_type.upper()})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'svm':
            estimator = SVR()
            search_space = {
                'C': Real(1e-2, 1e2, prior='log-uniform'),
                'epsilon': Real(1e-3, 1.0, prior='log-uniform'),
                'gamma': Real(1e-4, 1.0, prior='log-uniform'),
                'kernel': ['rbf']
            }
        else:
            estimator = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, verbosity=0)
            search_space = {
                'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
                'n_estimators': Integer(100, 600),
                'max_depth': Integer(2, 8),
                'subsample': Real(0.5, 1.0, prior='uniform'),
                'colsample_bytree': Real(0.5, 1.0, prior='uniform')
            }

        opt = BayesSearchCV(
            estimator,
            search_space,
            n_iter=30,
            cv=5,
            n_jobs=-1,
            random_state=42,
            scoring='neg_mean_squared_error'
        )
        
        opt.fit(X_train, y_train)
        
        print(f"  Best params: {opt.best_params_}")
        print(f"  Best CV MSE: {-opt.best_score_:.4f}")
        score = opt.score(X_test, y_test)
        print(f"  Test MSE: {-score:.4f}")

        # Save the best model
        model_path = f"models/{target_name.replace(' ','_').lower()}.joblib"
        joblib.dump(opt.best_estimator_, model_path)
        print(f"  Saved model to {model_path}")

@tool
def predict_properties(smiles: str):
    """
    Predicts properties for a given SMILES string using trained models.
    :param smiles: SMILES string
    :return: Dictionary with predictions
    """
    if not os.path.exists('models'):
        print("Error: 'models' directory not found. Please train the models first.")
        return {}

    # Generate descriptors
    desc = create_descriptor(smiles)
    # Check if descriptors were successfully generated (assuming create_descriptor returns None on failure or handles it internally)
    # Based on usage, it returns a list-like object.
    
    if desc is None:
        print("Error: Could not generate descriptors.")
        return {}

    # Reshape for prediction (1 sample, n features)
    X = np.array([desc])

    predictions = {}
    targets = [
        'Hf solid',
        'Det Velocity',
        'Det Pressure',
        'Density'
    ]

    for target_name in targets:
        model_filename = f"{target_name.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join('models', model_filename)
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # Check if model expects DMatrix (XGBoost sometimes does if not using sklearn API wrapper, 
                # but here we used XGBRegressor/SVR which use numpy arrays)
                pred = model.predict(X)[0]
                predictions[target_name] = float(pred)
            except Exception as e:
                print(f"Error predicting {target_name}: {e}")
                predictions[target_name] = None
        else:
            print(f"Warning: Model {model_filename} not found.")
            predictions[target_name] = None
            
    return predictions


def evaluate_models(df: pd.DataFrame):
    """
    Predicts properties for the full DataFrame and plots Q-Q residual plots.
    :param df: DataFrame containing SMILES and ground truth values
    :return: None
    """
    if not os.path.exists('models'):
        print("Error: 'models' directory not found. Please train the models first.")
        return

    # Generate descriptors if not present
    if 'desc' not in df.columns:
        print("Generating descriptors...")
        df['desc'] = df['SMILES'].apply(create_descriptor)
    
    # Save SMILES and descriptors to Excel
    print("Saving SMILES and descriptors to evaluated_molecules.xlsx...")
    df[['SMILES', 'desc']].astype(str).to_excel('evaluated_molecules.xlsx', index=False)
    
    X = np.array(df['desc'].tolist())
    
    targets = [
        'Hf solid',
        'Det Velocity',
        'Det Pressure',
        'Density'
    ]
    
    # Initialize figures for both plots
    fig_pred, axes_pred = plt.subplots(2, 2, figsize=(15, 10))
    axes_pred = axes_pred.flatten()
    
    fig_qq, axes_qq = plt.subplots(2, 2, figsize=(15, 10))
    axes_qq = axes_qq.flatten()

    for i, target_name in enumerate(targets):
        model_filename = f"{target_name.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join('models', model_filename)
        
        if os.path.exists(model_path):
            print(f"Evaluating {target_name}...")
            try:
                model = joblib.load(model_path)
                y_pred = model.predict(X)
                y_true = df[target_name].values
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = r2_score(y_true, y_pred)
                mean_expected = np.mean(y_true)
                print(f"{target_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}, Mean Expected: {mean_expected:.4f}")
                
                # Predicted vs Expected plot
                ax = axes_pred[i]
                ax.scatter(y_true, y_pred, alpha=0.5)
                
                # y=x line
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
                ax.plot(lims, lims, 'k--', alpha=0.75, zorder=1)
                
                ax.set_xlabel('Expected')
                ax.set_ylabel('Predicted')
                ax.set_title(f"{target_name}: Predicted vs Expected")
                ax.text(0.05, 0.95, f"RMSE: {rmse:.4f}\nR²: {r2:.4f}", 
                        transform=ax.transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                ax.grid(True)
                
                # Q-Q Residuals Plot
                residuals = y_pred - y_true
                # Handle cases with zero std to avoid division by zero
                residuals_std = np.std(residuals)
                if residuals_std != 0:
                    standardised_residuals = (residuals - np.mean(residuals)) / residuals_std
                else:
                    standardised_residuals = np.zeros_like(residuals)
                
                # Use statsmodels qqplot on the specific axis
                sm.qqplot(standardised_residuals, line='45', ax=axes_qq[i])
                axes_qq[i].set_title(f"{target_name}: Q-Q Plot")
                axes_qq[i].grid(True)

            except Exception as e:
                print(f"Error evaluating {target_name}: {e}")
        else:
            print(f"Warning: Model {model_filename} not found.")
            
    fig_pred.tight_layout()
    fig_pred.savefig(os.path.join('models', 'predicted_vs_expected.png'), dpi=300)
    plt.close(fig_pred)
    
    # Save Q-Q plots
    fig_qq.tight_layout()
    fig_qq.savefig(os.path.join('models', 'qq_residuals.png'), dpi=300)
    plt.close(fig_qq)


if __name__ == "__main__":
    df = import_data()
    train_models(df)
    evaluate_models(df)
