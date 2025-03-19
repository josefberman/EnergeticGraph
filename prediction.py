import os.path
import pickle

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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from langchain_core.tools import tool

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        y_train_mean = np.mean(y_train)
        # param_grid = {"alpha": np.logspace(-10, 2, 50), "gamma": np.logspace(-10, -1, 50), "kernel": ['rbf']}
        # grd_srch = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
        # param_grid = {'n_estimators': np.linspace(50, 500, 10, dtype=int), 'max_features': ['sqrt', 'log2', 1]}
        # grd_srch = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
        # param_grid = {'n_neighbors':np.arange(2,10,1)}
        # grd_srch = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
        param_grid = {'kernel': ['linear', 'poly', 'rbf'], 'degree': [2, 3, 4], 'C': np.logspace(-3, 2, 10),
                      'epsilon': np.logspace(-3, 2, 10)}
        grd_srch = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
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


# def train_data(energetic_property: str):
#     if not os.path.exists('./trained_models/'):
#         os.makedirs('./trained_models/')
#     df = pd.read_excel('clean_data_imputed.xlsx')
#     df.dropna(inplace=True)
#     X = df.iloc[:, :20]
#     property_dict = {'density': 20, 'gas phase formation enthalpy': 21, 'sublimation enthalpy': 22,
#                      'heat of explosion': 23, 'detonation velocity': 24, 'detonation pressure': 25,
#                      'gurney energy': 26, 'h50': 27}
#     y = df.iloc[:, property_dict[energetic_property]]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     y_train_mean = np.mean(y_train)
#     s_scaler = StandardScaler()
#     X_train = s_scaler.fit_transform(X_train)
#     X_test = s_scaler.transform(X_test)
#     # pca = PCA(n_components=4)
#     # X_train = pca.fit_transform(X_train)
#     # X_test = pca.transform(X_test)
#     # param_grid = {"alpha": np.logspace(-10, 2, 50), "gamma": np.logspace(-10, -1, 50), "kernel": ['rbf']}
#     # grd_srch = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring='neg_mean_square_error', verbose=0)
#     param_grid = {'n_estimators': np.linspace(50, 500, 10, dtype=int), 'max_features': ['sqrt', 'log2', 1]}
#     grd_srch = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
#     grd_srch.fit(X_train, y_train)
#     best_model = grd_srch.best_estimator_
#     with open(f'./trained_models/{energetic_property}.pkl', 'wb') as f:
#         pickle.dump(best_model, f)
#     # for param_name, param_value in grd_srch.best_params_.items():
#     # print(f'{param_name} : {param_value}')
#     # print('train r2 score:', r2_score(y_train, best_model.predict(X_train)))
#     # print('test r2 score:', r2_score(y_test, best_model.predict(X_test)))
#     # print('test rmse:', root_mean_squared_error(y_test, best_model.predict(X_test)))
#     # print('test rrmse:', root_mean_squared_error(y_test, best_model.predict(X_test)) / y_train_mean)
#     # print('test mae:', mean_absolute_error(y_test, best_model.predict(X_test)))
#     plt.figure(figsize=(5, 5))
#     y_pred = best_model.predict(X_test)
#     plt.scatter(y_test, y_pred, s=5, c='#e63946')
#     plt.plot([np.min([y_test, y_pred]), np.max([y_test, y_pred])], [np.min([y_test, y_pred]), np.max([y_test, y_pred])],
#              c='black', linewidth=1)
#     plt.xlabel('Test values')
#     plt.ylabel('Predicted values')
#     plt.title(energetic_property, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'./trained_models_plots/{energetic_property}.jpg', dpi=200)
#     plt.show()
#
#
# def train_data2(energetic_property: str):
#     if not os.path.exists('./trained_models/'):
#         os.makedirs('./trained_models/')
#     df = pd.read_excel('clean_data_imputed.xlsx')
#     df.dropna(inplace=True)
#     df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
#     df['new_desc'] = df['mol'].apply(lambda x: custom_descriptor_set(x))
#     X = pd.DataFrame(df['new_desc'].tolist())
#     property_dict = {'density': 20, 'gas phase formation enthalpy': 21, 'sublimation enthalpy': 22,
#                      'heat of explosion': 23, 'detonation velocity': 24, 'detonation pressure': 25,
#                      'gurney energy': 26, 'h50': 27}
#     y = df.iloc[:, property_dict[energetic_property]]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     y_train_mean = np.mean(y_train)
#     s_scaler = StandardScaler()
#     X_train = s_scaler.fit_transform(X_train)
#     X_test = s_scaler.transform(X_test)
#     # pca = PCA(n_components=4)
#     # X_train = pca.fit_transform(X_train)
#     # X_test = pca.transform(X_test)
#     # param_grid = {"alpha": np.logspace(-10, 2, 50), "gamma": np.logspace(-10, -1, 50), "kernel": ['rbf']}
#     # grd_srch = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=0)
#     param_grid = {'n_estimators': np.linspace(50, 500, 10, dtype=int), 'max_features': ['sqrt', 'log2', 1]}
#     grd_srch = GridSearchCV(RandomForestRegressor(), param_grid, cv=10, scoring='neg_mean_squared_error', verbose=0)
#     grd_srch.fit(X_train, y_train)
#     best_model = grd_srch.best_estimator_
#     with open(f'./trained_models/{energetic_property}.pkl', 'wb') as f:
#         pickle.dump(best_model, f)
#     # for param_name, param_value in grd_srch.best_params_.items():
#     #     print(f'{param_name} : {param_value}')
#     # print('train r2 score:', r2_score(y_train, best_model.predict(X_train)))
#     # print('test r2 score:', r2_score(y_test, best_model.predict(X_test)))
#     # print('test rmse:', root_mean_squared_error(y_test, best_model.predict(X_test)))
#     # print('test rrmse:', root_mean_squared_error(y_test, best_model.predict(X_test)) / y_train_mean)
#     # print('test mae:', mean_absolute_error(y_test, best_model.predict(X_test)))
#     plt.figure(figsize=(5, 5))
#     y_pred = best_model.predict(X_test)
#     plt.scatter(y_test, y_pred, s=5, c='#e63946')
#     plt.plot([np.min([y_test, y_pred]), np.max([y_test, y_pred])], [np.min([y_test, y_pred]), np.max([y_test, y_pred])],
#              c='black', linewidth=1)
#     plt.xlabel('Test values')
#     plt.ylabel('Predicted values')
#     plt.title(energetic_property, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'./trained_models_plots/{energetic_property}.jpg', dpi=200)
#     plt.show()


# def train_all_models():
#     properties = ['density', 'gas phase formation enthalpy', 'sublimation enthalpy', 'heat of explosion',
#                   'detonation velocity', 'detonation pressure', 'gurney energy', 'h50']
#     for p in properties:
#         train_data2(p)


@tool
def predict_properties(smiles: str) -> dict:
    """
    Predicts the following properties of energetic materials: Density, Gas phase formation enthalpy,
    sublimation enthalpy, heat of explosion, detonation velocity, detonation pressure, gurney energy and h50.
    :param smiles: SMILES string representing the molecule to predict.
    :return: Dictionary with predicted values
    """
    # descriptor = np.array(create_descriptor(smiles)).reshape(1,-1)
    descriptor = np.array(custom_descriptor_set(Chem.MolFromSmiles(smiles))).reshape(1, -1)
    print(descriptor)
    property_dict = {'density': 20, 'gas phase formation enthalpy': 21, 'sublimation enthalpy': 22,
                     'heat of explosion': 23, 'detonation velocity': 24, 'detonation pressure': 25,
                     'gurney energy': 26, 'h50': 27}
    predictions = {}
    for key in property_dict.keys():
        with open(f'./trained_models/{key}.pkl', 'rb') as f:
            model = pickle.load(f)
            predictions[key] = model.predict(descriptor)[0]
    return predictions


# def train_data_kernel_ridge(energetic_property: str):
#     df = pd.read_excel('clean_data.xlsx')
#     df.dropna(inplace=True)
#     X = df.iloc[:, :20]
#     property_dict = {'density': 20, 'gas phase formation enthalpy': 21, 'sublimation enthalpy': 22,
#                      'heat of explosion': 23, 'detonation velocity': 24, 'detonation pressure': 25,
#                      'gurney energy': 26, 'h50': 27}
#     y = df.iloc[:, property_dict[energetic_property]]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     y_train_mean = np.mean(y_train)
#     s_scaler = StandardScaler()
#     X_train = s_scaler.fit_transform(X_train)
#     X_test = s_scaler.transform(X_test)
#     param_grid = {'kernel': ['rbf', 'poly', 'linear'], 'alpha': np.logspace(-2, 2, 20, base=10),
#                   'gamma': np.logspace(-2, 2, 20, base=10),
#                   'degree': [1, 3, 5]}
#     grd_srch = GridSearchCV(KernelRidge(), param_grid, cv=10, scoring='r2')
#     grd_srch.fit(X_train, y_train)
#     best_model = grd_srch.best_estimator_
#     print(grd_srch.best_params_)
#     print('train r2 score:', r2_score(y_train, best_model.predict(X_train)))
#     print('test r2 score:', r2_score(y_test, best_model.predict(X_test)))
#     print('test rmse:', root_mean_squared_error(y_test, best_model.predict(X_test)))
#     print('test rrmse:', root_mean_squared_error(y_test, best_model.predict(X_test)) / y_train_mean)

df = pd.read_csv('extracted_chemical_data.csv')
train_data(df)
