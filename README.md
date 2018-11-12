# 2016_Data_Camp @ Ecole polytechnique

The Data camp was about two subjects (free choice):
- Drug : consists in classifying drugs and predicting their concentration in the solute
- Nino : consists in applying ARIMA to predict temperatures during el Nino event.

I worked on teh 'Drug' project.

## Context on the Drug project
Providing the right molecule at the right concentration is a crucial point for the chemiotherapy to be successful.
The aim of this challenge is to classify (4 classes) chemiotherapy molecules and predict their concentration based on 
their Raman spectra.


### Classification:
Predict the molecule present in the solution (Classes A, B, C, D)

### Regression:
Predict the concentration of the molecules

### Evaluation
0/1 lossfor classification, MARE for regression and a final combined error :
Combined: 2/3 * classifier + 1/3 * regressor



