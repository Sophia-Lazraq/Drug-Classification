import numpy as np
import pandas as pd
    
class FeatureExtractorClf(object):
    
    def __init__(self, window_length=11, polyorder=1):
        self.window_length = window_length
        self.polyorder = polyorder
 
    def fit(self, X_df, y_df):
        pass
    
    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        XX = np.log(XX)    
        return XX
