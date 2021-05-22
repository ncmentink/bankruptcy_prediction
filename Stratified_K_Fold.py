import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load the variables for logistic regression
from data_prep import X, y, X_smote, y_smote, X_smote_sc, y_smote_sc


skfold = StratifiedKFold(n_splits=4)
model_skfold = LogisticRegression(penalty='l1', solver='saga', max_iter=8000, C=100)
results_skfold = cross_val_score(model_skfold, X_smote_sc, y_smote_sc, cv=skfold)
print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
