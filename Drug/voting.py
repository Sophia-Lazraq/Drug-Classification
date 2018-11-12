import pandas as pd

np.random.seed(123)
clf1 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=200, 
                                       fit_intercept=True, intercept_scaling=1, class_weight=None, 
                                       random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', 
                                       verbose=0, warm_start=False, n_jobs=1)
clf2 = RandomForestClassifier(random_state=1) 
clf3 = SVC(C=9*10**5, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                    max_iter=-1, probability=True, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
clf4 = ExtraTreesClassifier() 
clf6 = GradientBoostingClassifier(max_depth=5,min_samples_split=4,learning_rate=0.3,max_features='log2',n_estimators=100,warm_start=True)
clf7=AdaBoostClassifier(RandomForestClassifier( max_depth=5,class_weight="balanced",n_estimators=38),n_estimators=38)
        
df = pd.DataFrame(columns=('w1', 'w2','w3', 'w6', 'w7','mean', 'std'))

i = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,6):
            for w6 in range(1,4):
                for w7 in range(1,4):

                    if len(set((w1,w2,w3,w6,w7))) == 1: # skip if all weights are equal
                        continue

                    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),('svc',clf3),('GradientBossting',clf6),('Adaboost',clf7)], voting='hard', weights=[w1,w2,w3,w6,w7])
                    scores = cross_val_score(
                                                estimator=eclf,
                                                X=X_pca,
                                                y=Y_lab,
                                                cv=5,
                                                scoring='accuracy',
                                                n_jobs=1)

                    df.loc[i] = [w1, w2,w3,w6,w7, scores.mean(), scores.std()]
                    i += 1

df.sort(columns=['mean', 'std'], ascending=False).head()
______________________________
from sklearn.decomposition import PCA                                            
from sklearn.pipeline import Pipeline                                            
from sklearn.base import BaseEstimator                                           
import numpy as np                                                               
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
def mare_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
mare_scorer = make_scorer(mare_score,greater_is_better=False)

class Regressor(BaseEstimator):                                                  
    def __init__(self):                                                          
        self.n_components = 11                                                
        self.list_molecule = ['A', 'B', 'Q', 'R']                                
        self.dict_reg = {}                                                       
        for mol in self.list_molecule:                                           
            self.dict_reg[mol] = Pipeline([
                ('pca', PCA(n_components=self.n_components)),                     
                ('svr', GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e1, 1e2, 1e3],
                               "gamma": [1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3]},scoring=mare_scorer))
            ])                                                                   
                                                                                 
    def fit(self, X, y):                                                         
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol]                                                  
            y_mol = y[ind_mol].astype(float)                                     
            self.dict_reg[mol].fit(XX_mol, np.log(y_mol)) 
                              
                                                                                 
    def predict(self, X):                                                        
        y_pred = np.zeros(X.shape[0])                                            
        for i, mol in enumerate(self.list_molecule):                             
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol].astype(float)                                    
            y_pred[ind_mol] = np.exp(self.dict_reg[mol].predict(XX_mol))         
        return y_pred  

___________________________________
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
        XX= savgol_filter(XX, 51, 9)

        return XX   
