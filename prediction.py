import os.path
import pickle
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from langchain_core.tools import tool
import statsmodels.api as sm

from descriptors import custom_descriptor_set


def get_nno2_count(smiles: str):
    """
    Counts nitrogen-nitro group
    :param smiles: Molecule in SMILES format
    :return: Number of nitrogen-nitro groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7][#7+](=[#8])[#8-]')))


def get_cno2_count(smiles: str):
    """
    Counts nitro group
    :param smiles: Molecule in SMILES format
    :return: Number of nitro groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7+](=[#8])[#8-]')))


def get_ono2_count(smiles: str):
    """
    Counts oxygen-nitro group
    :param smiles: Molecule in SMILES format
    :return: Number of oxygen-nitro groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8][#7+](=[#8])[#8-]')))


def get_ono_count(smiles: str):
    """
    Counts nitrite group
    :param smiles: Molecule in SMILES format
    :return: Number of nitrite groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8][#7]=[#8]')))


def get_cno_count(smiles: str):
    """
    Counts fulminate group with tautomers
    :param smiles: Molecule in SMILES format
    :return: Number of fulminate groups (with tautomers) within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#7][#8]'))) + len(
        mol.GetSubstructMatches(Chem.MolFromSmarts('[#6-]#[#7+][#8]')))


def get_cnn_count(smiles: str):
    """
    Counts azo group
    :param smiles: Molecule in SMILES format
    :return: Number of azo groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7]=[#7]')))


def get_nnn_count(smiles: str):
    """
    Counts azide group
    :param smiles: Molecule in SMILES format
    :return: Number of azide groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#7+]=[#7-]'))) + len(
        mol.GetSubstructMatches(Chem.MolFromSmarts('[#7-][#7+]#[#7]')))


def get_cnh2_count(smiles: str):
    """
    Counts amine group
    :param smiles: Molecule in SMILES format
    :return: Number of amine groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7]([#1])[#1]')))


def get_cnoc_count(smiles: str):
    """
    Counts N-oxide nitrogen group
    :param smiles: Molecule in SMILES format
    :return: Number of N-oxide nitrogen groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7]([#8])[#6]')))


def get_cnf_count(smiles: str):
    """
    Counts nitrogen-fluorine group
    :param smiles: Molecule in SMILES format
    :return: Number of nitrogen-fluorine groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#7][#9]')))


def get_c_count(smiles: str):
    """
    Counts carbons
    :param smiles: Molecule in SMILES format
    :return: Number of carbons within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]')))


def get_n_count(smiles: str):
    """
    Counts nitrogens
    :param smiles: Molecule in SMILES format
    :return: Number of nitrogens within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]')))


def get_h_count(smiles: str):
    """
    Counts hydrogens
    :param smiles: Molecule in SMILES format
    :return: Number of hydrogens within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#1]')))


def get_f_count(smiles: str):
    """
    Counts fluorides
    :param smiles: Molecule in SMILES format
    :return: Number of fluorides within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#9]')))


def get_no_count(smiles: str):
    """
    Counts nitrate/nitrite
    :param smiles: Molecule in SMILES format
    :return: Number of nitrate/nitrite groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#8]')))


def get_co_count(smiles: str):
    """
    Counts carbonyl/keton
    :param smiles: Molecule in SMILES format
    :return: Number of carbonyl/keton groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]')))


def get_coh_count(smiles: str):
    """
    Counts hydroxyl
    :param smiles: Molecule in SMILES format
    :return: Number of hydroxyl groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#8][#1]')))


def get_noc_count(smiles: str):
    """
    Counts N-oxide oxygen groups
    :param smiles: Molecule in SMILES format
    :return: Number of N-oxide oxygen groups within the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7][#8][#6]')))


def calc_ob_100(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    n_O = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8]')))
    n_C = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]')))
    n_H = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#1]')))
    n_atoms = mol.GetNumAtoms()
    return 100 / n_atoms * (n_O - 2 * n_C - n_H / 2)


def get_n_over_c(smiles: str):
    return get_n_count(smiles) / get_c_count(smiles)


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


def train_data(df: pd.DataFrame):
    df['desc'] = df['SMILES'].apply(create_descriptor)
    X = np.array(df['desc'].tolist())
    for col in df.iloc[:, :-2].columns:
        y = df[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        with open('./trained_models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        y_train_mean = np.mean(y_train)
        param_grid = {"alpha": np.logspace(-10, 2, 50), "gamma": np.logspace(-10, -1, 50), "kernel": ['rbf']}
        grd_srch = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
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
    Predicts the following properties of energetic materials: Density, Solid-phase formation enthalpy,
    Detonation speed, Explosion capacity, Explosion pressure, Explosive heat.
    :param smiles: SMILES string representing the molecule to predict.
    :return: Dictionary with predicted values
    """
    descriptor = np.array(create_descriptor(smiles)).reshape(1, -1)
    with open('./trained_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    descriptor = scaler.transform([descriptor])
    property_list = ['Density', 'Detonation velocity', 'Explosion capacity', 'Explosion pressure', 'Explosion heat',
                     'Solid phase formation enthalpy']
    predictions = {}
    for key in property_list:
        with open(f'./trained_models/{key}.pkl', 'rb') as f:
            model = pickle.load(f)
            predictions[key] = model.predict(descriptor)[0]
    return predictions
