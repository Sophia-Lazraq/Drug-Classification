import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
 
class Regressor(BaseEstimator):
    def __init__(self, n_components=[14, 15, 10, 10], gamma=[2, 1, 1, 2], C=[30000, 30000, 30000, 100000]):
        self.n_components = n_components
        self.list_molecule = ['A', 'B', 'Q', 'R']
        self.dict_reg_svc = {}
        for i, mol in enumerate(self.list_molecule):
            self.dict_reg_svc[mol] = Pipeline([
                ('kpca', KernelPCA(n_components=self.n_components[i], fit_inverse_transform=True,
                                   eigen_solver='arpack')),
                ('svc', SVC(C=C[i], kernel='poly', degree=4, gamma=gamma[i], coef0=2, probability=True))
            ])
 
    def fit(self, X, y):
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            XX_mol = X[ind_mol]
            y_mol = y[ind_mol].astype(float)
            self.dict_reg_svc[mol].fit(XX_mol, y_mol)
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            XX_mol = X[ind_mol].astype(float)
            y_pred[ind_mol] = self.dict_reg_svc[mol].predict(XX_mol)
        return y_pred