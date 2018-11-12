from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np    
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
 
class Classifier(BaseEstimator):
    
    def __init__(self):
        
        ####################################
        # Classifier 1 : Gradient Boosting #
        ####################################
        
        self.n_components_gb = 10
        self.loss_gb = "deviance"
        self.n_estimators_gb = 50
        self.learning_rate_gb = 0.4
        
        self.clf_gb = Pipeline([
            ('pca', PCA(n_components=self.n_components_gb)),
            ('clf', GradientBoostingClassifier(
                        loss = self.loss_gb, 
                        learning_rate = self.learning_rate_gb,
                        n_estimators = self.n_estimators_gb, 
                        random_state = None))
        ])
 
 
        
        #################################
        # Classifier 2 : Neural Network #
        #################################
        
        self.n_components_mlp = 17
        self.activation_mlp = "tanh"
        self.hidden_layer_sizes_mlp = (100,100)
        self.alpha_mlp = 0.0005
        
        self.clf_mlp = Pipeline([
            ('pca', PCA(n_components = self.n_components_mlp)),
            ('clf', MLPClassifier(
                        activation = self.activation_mlp,
                        hidden_layer_sizes = self.hidden_layer_sizes_mlp, 
                        alpha = self.alpha_mlp))
        ])

        self.n_components_svc = 10
#        param_grid = {'C':[9*1e5,1e6,1e7],'gamma':np.logspace(-3,3,7)}
        self.clf_svc = Pipeline([
            ('pca', PCA(n_components=self.n_components_svc)), 
            ('SVC', SVC(probability=True,C=9e5,gamma=0.1))
# GridSearchCV(SVC(probability=True),param_grid=param_grid,scoring='accuracy')                 
            ]) 
        
        
        ####################
        # VotingClassifier #
        ####################
        
        self.clf = VotingClassifier(
            estimators = [('gb', self.clf_gb), 
                          ('mlp', self.clf_mlp),
                          ('svc', self.clf_svc)],
            voting = 'soft',
            weights=[0.1,0.5,0.4])
 
    
    def fit(self, X, y):
        self.clf.fit(X,y)
 
    def predict(self, X): 
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)