import numpy as np
 
class FeatureExtractorReg(object):
    def __init__(self, window=15):
        self.window = window
        self.list_molecule = ['A', 'B', 'Q', 'R']
 
    def fit(self, X_df, y):
        pass
    
    def smooth(self, spectrum, window_len=11):
        spectrum_extract = np.r_[spectrum[window_len-1:0:-1], spectrum, spectrum[-1:-window_len:-1]]
        hanning_window = np.hanning(window_len)
        smooth_spectrum = np.convolve(hanning_window / hanning_window.sum(), spectrum_extract, mode='valid')
        return smooth_spectrum[(self.window - 1) // 2:(1 - self.window) // 2]
            
    def transform(self, X_df):
        XX = np.array([
            self.smooth(np.array(dd), self.window) for dd in X_df['spectra']
        ])                  
        XX -= np.median(XX, axis=1)[:, None]
        XX /= np.sqrt(np.sum(XX ** 2, axis=1))[:, None]
        XX = np.hstack((XX, X_df[self.list_molecule].values))
        return XX