from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

# Load the variables for logistic regression
from data_prep import X, y, X_smote, y_smote, X_smote_sc, y_smote_sc

"""
#lasso smote sc
skfold = StratifiedKFold(n_splits=4)
model_skfold = LogisticRegression(max_iter=8000)
results_skfold = cross_val_score(model_skfold, X_smote_sc, y_smote_sc, cv=skfold)
print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))


#lasso unchanged data
skfold = StratifiedKFold(n_splits=4)
model_skfold = LogisticRegression(penalty='l1', solver='lbfgs', max_iter=8000, C=1000)
results_skfold = cross_val_score(model_skfold, X, y, cv=skfold)
print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
"""

# First we scale the data, as this is necessary for Ridge and Lasso
# For the Lasso, the regularization penalty is comprised of the sum of the absolute value of the coefficients,
# therefore we need to scale the data so the coefficients are all based on the same scale.
X = StandardScaler().fit_transform(X)

# Pick model
# Lasso
# model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000)

# Ridge
model = LogisticRegression(penalty="l2", solver='lbfgs', max_iter=8000)


# Pick performance measure
# measure = "precision"
measure = "accuracy"
# measure = "recall"
# measure = "roc_auc"


# 1: decision tree evaluated on imbalanced dataset
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring=measure, cv=cv, n_jobs=-1)
print(measure, '%.3f' % mean(scores))


# 2: decision tree evaluated on imbalanced dataset with SMOTE oversampling
# define pipeline
steps = [('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
print(measure, '%.3f' % mean(scores))


# 3: decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
# define pipeline
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=measure, cv=cv, n_jobs=-1)
    score = mean(scores)
    print(measure, '> k=%d %.3f' % (k, score))
