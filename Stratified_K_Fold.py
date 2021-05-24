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

"""
# First we scale the data, as this is necessary for Ridge and Lasso
# For the Lasso, the regularization penalty is comprised of the sum of the absolute value of the coefficients,
# therefore we need to scale the data so the coefficients are all based on the same scale.
X = StandardScaler().fit_transform(X)


# Pick model
# Lasso
model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000)

# Ridge
#model = LogisticRegression(penalty="l2", solver='lbfgs', max_iter=8000)

# Pick performance measure to compare models on
# ROC AUC is best to compare different datasets??
# measure = "precision"
measure = "accuracy"
# measure = "recall"
# measure = "roc_auc"


# The folds of the cross-validation split are stratified, which means they will have the same class distribution as the
# original dataset, in this case a 1:32 ratio (3% 1-class, 97 0-class) and later a 1:2 ratio (33% 1-class, 66% 0-class).

# 1: decision tree evaluated on imbalanced dataset
# evaluate pipeline
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, scoring=measure, cv=cv, n_jobs=-1)
print(measure, '%.3f' % mean(scores))


# 2: decision tree evaluated on imbalanced dataset with SMOTE oversampling
# define pipeline
steps = [('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
print(measure, '%.3f' % mean(scores))


# 3: decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
# Define pipeline
# Over sample to a 1:10 ration
over = SMOTE(sampling_strategy=0.1)
# Under sample to a 1:2 ratio
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
print(measure, '%.3f' % mean(scores))


# 4: grid search k value for SMOTE oversampling for imbalanced classification
# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]

for k in k_values:
    # define pipeline
    over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    cv = StratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print(measure, '> k=%d %.3f' % (k, score))
"""




from sklearn.metrics import classification_report, make_scorer, accuracy_score, f1_score, recall_score

def classification_report_with_f1_score(y_true, y_pred):
    true_labels.extend(y_true)
    predicted_labels.extend(y_pred)

    return f1_score(y_true, y_pred)


# C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions.
C = [10, 5, 1, 0.5, .1, 0.05, .001]

for c in C:
    # Lasso
    model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000)

    # Ridge
    #model = LogisticRegression(penalty="l2", solver='lbfgs', max_iter=8000)

    # Variables for average classification report
    true_labels = []
    predicted_labels = []

    # Over sample to a 1:10 ration
    over = SMOTE(sampling_strategy=0.1)

    # Under sample to a 1:2 ratio
    under = RandomUnderSampler(sampling_strategy=0.5)

    # Pipelines help avoid leaking statistics from your test data into the trained model in cross-validation,
    # by ensuring that the same samples are used to train the transformers and predictors. Also, the scaler is fit on
    # the training data, transforms the train data, models are fitted on the train data, and the scaler is used to
    # transform the test data. Therefore, the test data is not used to determine the scaling parameters.
    steps = [('over', over), ('under', under), ('scale', StandardScaler()), ('model', model)]
    pipeline = Pipeline(steps=steps)

    # Nested CV with parameter optimization (f1_score)
    cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
                    scoring=make_scorer(classification_report_with_f1_score))

    # Average values in classification report for all folds in a K-fold Cross-validation
    print('C:', c)
    print('model:', model.penalty)
    print(classification_report(true_labels, predicted_labels))
