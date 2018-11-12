import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class FeatureExtractorClf(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_df):
        pass
    
    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        XX= savgol_filter(XX, 101, 11)

        return XX
____________________


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor ,ExtraTreesRegressor                          
from sklearn.decomposition import PCA ,KernelPCA                                           
from sklearn.pipeline import Pipeline                                            
from sklearn.base import BaseEstimator                                           
import numpy as np     
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 11
        self.n_estimators = 300
        #param_grid = {'C':[9*10**5,10**6,10**7],'gamma':np.logspace(-3,3,7)}
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)), 
            ('SVC', SVC(C=9*10**5,probability=True))])       
             
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
____________________
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import savgol_filter


labels = np.array(['A', 'B', 'Q', 'R'])
class FeatureExtractorReg(object):
    def __init__(self):
        self.clf = StandardScaler()

    def fit(self, X_df, y_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        self.clf.fit(XX)
    
    def transform(self, X_df):   
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        XX = self.clf.transform(XX)
        XX = np.concatenate([XX, X_df[labels].values], axis=1) 
        XX= savgol_filter(XX, 101, 11)                                                

        return XX
        _____________________
                                
from sklearn.decomposition import PCA                                            
from sklearn.pipeline import Pipeline                                            
from sklearn.base import BaseEstimator                                           
import numpy as np 
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

    
def mare_score(y_true, y_pred):                                                  
    return np.mean(np.abs((y_true - y_pred) / y_true))
    
mare_scorer = make_scorer(mare_score,greater_is_better=False)

class Regressor(BaseEstimator):                                                  
    def __init__(self):                                                          
        self.n_components = 11                                                                                                                                                     
        self.list_molecule = ['A', 'B', 'Q', 'R']
        self.dict_reg = {}
        param_grid = {'C':[1,10**2,10**3,10**4,10**5,10**6],'gamma':np.logspace(-3,3,7)}

        for mol in self.list_molecule:                                           
            self.dict_reg[mol] = Pipeline([                                      
                ('pca', PCA(n_components=self.n_components)),                    
                ('SVR', GridSearchCV(estimator=SVR(), 
            param_grid=param_grid, scoring=mare_scorer, fit_params=None, n_jobs=1, iid=True, 
            refit=True, cv=None, verbose=0, 
            pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True))                                           
            ])                                                                   
                                                                                 
    def fit(self, X, y):                                                         
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol]                                                  
            y_mol = y[ind_mol].astype(float)                                     
            self.dict_reg[mol].fit(XX_mol, np.log(y_mol))
            print(self.dict_reg[mol].named_steps['SVR'].best_params_)
                                                                                
    def predict(self, X):                                                        
        y_pred = np.zeros(X.shape[0])                                            
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol].astype(float)                                    
            y_pred[ind_mol] = np.exp(self.dict_reg[mol].predict(XX_mol))         
        return y_pred   



