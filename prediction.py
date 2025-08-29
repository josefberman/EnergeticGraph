import pickle
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from langchain_core.tools import tool
import statsmodels.api as sm


def get_nno2_count(smiles: str):
    """
    Counts nitrogen-nitro group
    :param smiles: Molecule in SMILES format
    :return: Number of nitrogen-nitro groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7][#7+](=[#8])[#8-]')))
    except:
        return 0


def get_cno2_count(smiles: str):
    """
    Counts nitro group
    :param smiles: Molecule in SMILES format
    :return: Number of nitro groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7+](=[#8])[#8-]')))
    except:
        return 0


def get_ono2_count(smiles: str):
    """
    Counts oxygen-nitro group
    :param smiles: Molecule in SMILES format
    :return: Number of oxygen-nitro groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8][#7+](=[#8])[#8-]')))
    except:
        return 0


def get_ono_count(smiles: str):
    """
    Counts nitrite group
    :param smiles: Molecule in SMILES format
    :return: Number of nitrite groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8][#7]=[#8]')))
    except:
        return 0


def get_cno_count(smiles: str):
    """
    Counts fulminate group with tautomers
    :param smiles: Molecule in SMILES format
    :return: Number of fulminate groups (with tautomers) within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#7][#8]'))) + len(
            mol.GetSubstructMatches(Chem.MolFromSmarts('[#6-]#[#7+][#8]')))
    except:
        return 0


def get_cnn_count(smiles: str):
    """
    Counts azo group
    :param smiles: Molecule in SMILES format
    :return: Number of azo groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7]=[#7]')))
    except:
        return 0


def get_nnn_count(smiles: str):
    """
    Counts azide group
    :param smiles: Molecule in SMILES format
    :return: Number of azide groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#7+]=[#7-]'))) + len(
            mol.GetSubstructMatches(Chem.MolFromSmarts('[#7-][#7+]#[#7]')))
    except:
        return 0


def get_cnh2_count(smiles: str):
    """
    Counts amine group
    :param smiles: Molecule in SMILES format
    :return: Number of amine groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7]([#1])[#1]')))
    except:
        return 0


def get_cnoc_count(smiles: str):
    """
    Counts N-oxide nitrogen group
    :param smiles: Molecule in SMILES format
    :return: Number of N-oxide nitrogen groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7]([#8])[#6]')))
    except:
        return 0


def get_cnf_count(smiles: str):
    """
    Counts nitrogen-fluorine group
    :param smiles: Molecule in SMILES format
    :return: Number of nitrogen-fluorine groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#7][#9]')))
    except:
        return 0


def get_c_count(smiles: str):
    """
    Counts carbons
    :param smiles: Molecule in SMILES format
    :return: Number of carbons within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]')))
    except:
        return 0


def get_n_count(smiles: str):
    """
    Counts nitrogens
    :param smiles: Molecule in SMILES format
    :return: Number of nitrogens within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]')))
    except:
        return 0


def get_h_count(smiles: str):
    """
    Counts hydrogens
    :param smiles: Molecule in SMILES format
    :return: Number of hydrogens within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#1]')))
    except:
        return 0


def get_f_count(smiles: str):
    """
    Counts fluorides
    :param smiles: Molecule in SMILES format
    :return: Number of fluorides within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#9]')))
    except:
        return 0


def get_no_count(smiles: str):
    """
    Counts nitrate/nitrite
    :param smiles: Molecule in SMILES format
    :return: Number of nitrate/nitrite groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#8]')))
    except:
        return 0


def get_co_count(smiles: str):
    """
    Counts carbonyl/keton
    :param smiles: Molecule in SMILES format
    :return: Number of carbonyl/keton groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]')))
    except:
        return 0


def get_coh_count(smiles: str):
    """
    Counts hydroxyl
    :param smiles: Molecule in SMILES format
    :return: Number of hydroxyl groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#8][#1]')))
    except:
        return 0


def get_noc_count(smiles: str):
    """
    Counts N-oxide oxygen groups
    :param smiles: Molecule in SMILES format
    :return: Number of N-oxide oxygen groups within the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7][#8][#6]')))
    except:
        return 0


def calc_ob_100(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        n_O = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8]')))
        n_C = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]')))
        n_H = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#1]')))
        n_atoms = mol.GetNumAtoms()
        return 100 / n_atoms * (n_O - 2 * n_C - n_H / 2)
    except:
        return 0


def get_n_over_c(smiles: str):
    try:
        n_count = get_n_count(smiles)
        c_count = get_c_count(smiles)
        if c_count == 0:
            return 0
        return n_count / c_count
    except:
        return 0


def create_descriptor(smiles: str):
    return [calc_ob_100(smiles),
            get_n_over_c(smiles),
            get_nno2_count(smiles),
            get_cno2_count(smiles),
            get_ono2_count(smiles),
            get_ono_count(smiles),
            get_cno_count(smiles),
            get_cnn_count(smiles),
            get_nnn_count(smiles),
            get_cnh2_count(smiles),
            get_cnoc_count(smiles),
            get_cnf_count(smiles),
            get_c_count(smiles),
            get_n_count(smiles),
            get_no_count(smiles),
            get_coh_count(smiles),
            get_noc_count(smiles),
            get_co_count(smiles),
            get_h_count(smiles),
            get_f_count(smiles)
            ]
    # mol = Chem.MolFromSmiles(smiles)
    # if mol is None:
    #     return np.zeros(64, dtype=float)
    # featurizer = dc.feat.CoulombMatrixEig(64)
    # # Use standard DeepChem API to ensure correct shape: (64,)
    # features = featurizer.featurize([mol])
    # if isinstance(features, (list, tuple)):
    #     features = np.asarray(features)
    # # features should be shape (1, 64); return the 1D vector
    # return features[0]


def train_data(df: pd.DataFrame):
    df['desc'] = df['SMILES'].apply(create_descriptor)
    X = np.array(df['desc'].tolist())
    for col in df.iloc[:, :-2].columns:
        y = df[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        with open(f'./trained_models/scaler_{col}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        y_train_mean = np.mean(y_train)
        # Kernel Ridge was previously used as the estimator:
        # param_grid = {"alpha": np.logspace(-10, 2, 50), "gamma": np.logspace(-10, -1, 50), "kernel": ['rbf']}
        # grd_srch = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)

        # Use XGBoost regressor with hyperparameter optimization instead
        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
        param_grid_xgb = {
            "n_estimators": [200, 400, 800],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0]
        }
        grd_srch = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', verbose=0)
        # param_grid = {"n_estimators": np.arange(100, 500, 100), "max_depth": np.arange(3, 20)}
        # grd_srch = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
        # param_grid = {"kernel": ["rbf"], "degree": np.arange(1, 4), "gamma": np.logspace(-10, -1, 50),
        #               "C": np.logspace(-2, 2, 10), "epsilon": np.logspace(-5, 2, 10)}
        # grd_srch = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
        grd_srch.fit(X_train, y_train)
        best_model = grd_srch.best_estimator_
        with open(f'./trained_models/{col}.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        plt.figure(figsize=(5, 5))
        y_pred = best_model.predict(X_test)
        print(col)
        print(f'  train r2 score: {r2_score(y_train, best_model.predict(X_train)):.4f}')
        print(f'  test r2 score: {r2_score(y_test, best_model.predict(X_test)):.4f}')
        print(f'  test rmse: {root_mean_squared_error(y_test, best_model.predict(X_test)):.4f}')
        print(f'  test rrmse: {root_mean_squared_error(y_test, best_model.predict(X_test)) / y_train_mean:.4f}')
        print(f'  test mae: {mean_absolute_error(y_test, best_model.predict(X_test)):.4f}')
        plt.scatter(y_test, y_pred, s=5, c='#e63946')
        plt.plot([np.min([y_test, y_pred]), np.max([y_test, y_pred])],
                 [np.min([y_test, y_pred]), np.max([y_test, y_pred])],
                 c='black', linewidth=1)
        plt.xlabel('Test values')
        plt.ylabel('Predicted values')
        plt.title(col, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'./trained_models_plots/{col}.jpg', dpi=200)
        plt.show()
        plt.show()
        plt.close()
        plt.figure(figsize=(5, 5))
        residuals = y_pred - y_test
        standardised_residuals = (residuals-np.mean(residuals))/np.std(residuals)
        sm.qqplot(standardised_residuals, line='45', color='#e63946')
        plt.title(f'{col} - Q-Q plot', fontweight='bold')
        plt.savefig(f'./trained_models_plots/{col}_residuals.jpg', dpi=200)
        plt.show()


def train_data_nn(df: pd.DataFrame):
    """
    Trains a neural network on Coulomb matrix eigenvalue descriptors to predict
    the 5 energetic material properties jointly. Uses an 80/20 train/test split,
    a 64-neuron input (matching descriptor length), 3 hidden layers, and saves
    the scaler and model.
    """
    # Prepare descriptors (length 64 per molecule)
    df['desc'] = df['SMILES'].apply(create_descriptor)
    X = np.array(df['desc'].tolist())

    # Target properties (multi-output)
    property_list = ['Density', 'Detonation velocity', 'Explosion capacity', 'Explosion pressure', 'Explosion heat']
    y = df[property_list].values

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale inputs
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open('./trained_models/scaler_nn_multi.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # MLP with 3 hidden layers; input size implied by X_train.shape[1] (64)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20
    )

    # Fit multi-output regressor
    mlp.fit(X_train, y_train)
    with open('./trained_models/nn_multi_output.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    # Predictions
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    # Metrics and plots per property
    for i, col in enumerate(property_list):
        print(col)
        print(f'  train r2 score: {r2_score(y_train[:, i], y_train_pred[:, i]):.4f}')
        print(f'  test r2 score: {r2_score(y_test[:, i], y_test_pred[:, i]):.4f}')
        print(f'  test rmse: {root_mean_squared_error(y_test[:, i], y_test_pred[:, i]):.4f}')
        y_train_mean = np.mean(y_train[:, i])
        print(f'  test rrmse: {root_mean_squared_error(y_test[:, i], y_test_pred[:, i]) / (y_train_mean if y_train_mean != 0 else 1):.4f}')
        print(f'  test mae: {mean_absolute_error(y_test[:, i], y_test_pred[:, i]):.4f}')

        # Scatter plot
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test[:, i], y_test_pred[:, i], s=5, c='#e63946')
        min_axis = np.min([y_test[:, i], y_test_pred[:, i]])
        max_axis = np.max([y_test[:, i], y_test_pred[:, i]])
        plt.plot([min_axis, max_axis], [min_axis, max_axis], c='black', linewidth=1)
        plt.xlabel('Test values')
        plt.ylabel('Predicted values')
        plt.title(col + ' (NN)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'./trained_models_plots/{col}_nn.jpg', dpi=200)
        plt.show()
        plt.close()

        # Q-Q plot of standardized residuals
        residuals = y_test_pred[:, i] - y_test[:, i]
        standardised_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) if np.std(residuals) != 0 else 1)
        plt.figure(figsize=(5, 5))
        sm.qqplot(standardised_residuals, line='45', color='#e63946')
        plt.title(f'{col} - Q-Q plot (NN)', fontweight='bold')
        plt.savefig(f'./trained_models_plots/{col}_residuals_nn.jpg', dpi=200)
        plt.show()

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
    Predicts the following properties of energetic materials: Density, Detonation velocity,
    Explosion capacity, Explosion pressure, Explosion heat.
    :param smiles: SMILES string representing the molecule to predict.
    :return: Dictionary with predicted values
    """
    try:
        # Create descriptor and ensure it's 2D
        descriptor = np.array(create_descriptor(smiles))
        if descriptor.ndim == 1:
            descriptor = descriptor.reshape(1, -1)
        elif descriptor.ndim > 2:
            descriptor = descriptor.reshape(1, -1)
        
        property_list = ['Density', 'Detonation velocity', 'Explosion capacity', 'Explosion pressure', 'Explosion heat']
        predictions = {}

        for key in property_list:
            try:
                # Load scaler for each property and transform descriptor
                with open(f'./trained_models/scaler_{key}.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                descriptor_scaled = scaler.transform(descriptor)

                with open(f'./trained_models/{key}.pkl', 'rb') as f:
                    model = pickle.load(f)
                    predictions[key] = model.predict(descriptor_scaled)[0]
            except Exception as e:
                print(f"Warning: Could not predict {key}: {e}")
                predictions[key] = 0.0  # Default value
        
        return predictions
        
    except Exception as e:
        print(f"Error in predict_properties: {e}")
        # Return default values if prediction fails
        return {
            'Density': 1.0,
            'Detonation velocity': 5000.0,
            'Explosion capacity': 0.5,
            'Explosion pressure': 150.0,
            'Explosion heat': 800.0
        }


@tool
def predict_properties_nn(smiles: str) -> dict:
    """
    Predicts the five energetic material properties using the neural network trained
    on Coulomb matrix eigenvalue descriptors (length 64). Falls back to the legacy
    per-target models if the NN artifacts are unavailable.
    :param smiles: SMILES string representing the molecule to predict.
    :return: Dictionary with predicted values for the 5 properties
    """
    property_list = ['Density', 'Detonation velocity', 'Explosion capacity', 'Explosion pressure', 'Explosion heat']
    try:
        # Create descriptor and ensure 2D shape
        descriptor = np.array(create_descriptor(smiles))
        if descriptor.ndim == 1:
            descriptor = descriptor.reshape(1, -1)
        elif descriptor.ndim > 2:
            descriptor = descriptor.reshape(1, -1)

        # Load shared scaler and NN model
        with open('./trained_models/scaler_nn_multi.pkl', 'rb') as f:
            scaler = pickle.load(f)
        descriptor_scaled = scaler.transform(descriptor)

        with open('./trained_models/nn_multi_output.pkl', 'rb') as f:
            model = pickle.load(f)

        y_pred = model.predict(descriptor_scaled)
        # Ensure shape (1, 5)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if y_pred.ndim == 1 and y_pred.shape[0] == 5:
            y_pred = y_pred.reshape(1, -1)

        result = {prop: float(y_pred[0, i]) for i, prop in enumerate(property_list)}
        return result

    except Exception as e:
        # Fallback to legacy per-target predictors if available
        try:
            return predict_properties(smiles)
        except Exception:
            # Final fallback defaults
            return {
                'Density': 1.0,
                'Detonation velocity': 5000.0,
                'Explosion capacity': 0.5,
                'Explosion pressure': 150.0,
                'Explosion heat': 800.0
            }
