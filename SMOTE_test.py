import numpy as np
import pandas as pd
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("data.csv")

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# Pick model
# Lasso
model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000)

# Ridge
#model = LogisticRegression(penalty="l2", solver='lbfgs', max_iter=8000)

# Pick performance measure to compare models on
# ROC AUC is best to compare different datasets??
# measure = "precision"
# measure = "accuracy"
# measure = "recall"
measure = "roc_auc"


# 1: decision tree evaluated on imbalanced dataset
# evaluate pipeline
steps = [('scaler', StandardScaler()), ('model', model)]
pipeline = Pipeline(steps=steps)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
print("Model", model.penalty)
print(measure, '%.3f' % mean(scores))


# 2: decision tree evaluated on imbalanced dataset with SMOTE oversampling
# define pipeline
steps = [('scaler', StandardScaler()), ('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
print("Model", model.penalty)
print(measure, '%.3f' % mean(scores))


# 3: decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
# Define pipeline
# Over sample to a 1:10 ration
over = SMOTE(sampling_strategy=0.1)
# Under sample to a 1:2 ratio
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('scaler', StandardScaler()),('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
print("Model", model.penalty)
print(measure, '%.3f' % mean(scores))


# 4: grid search k value for SMOTE oversampling for imbalanced classification
# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]

for k in k_values:
    # define pipeline
    over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('scaler', StandardScaler()),('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    cv = StratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print("Model", model.penalty)
    print(measure, '> k=%d %.3f' % (k, score))
