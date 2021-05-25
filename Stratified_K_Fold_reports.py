from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import classification_report, accuracy_score, make_scorer
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
measure1 = "precision"
measure2 = "accuracy"
measure3 = "recall"
measure4 = "roc_auc"


# 1: decision tree evaluated on imbalanced dataset
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores1 = cross_val_score(model, X, y, scoring=measure1, cv=cv, n_jobs=-1)
scores2 = cross_val_score(model, X, y, scoring=measure2, cv=cv, n_jobs=-1)
scores3 = cross_val_score(model, X, y, scoring=measure3, cv=cv, n_jobs=-1)
scores4 = cross_val_score(model, X, y, scoring=measure4, cv=cv, n_jobs=-1)
print(measure1, '%.3f' % mean(scores1),'\n',measure2, '%.3f' % mean(scores2),'\n',
      measure3, '%.3f' % mean(scores3),'\n',measure4, '%.3f' % mean(scores4))

exit()


def classification_report_with_accuracy_score(y_true, y_pred):

    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Nested CV with parameter optimization
scores = cross_val_score(model, X, y,scoring=make_scorer(classification_report_with_accuracy_score), cv=cv, n_jobs=-1)

print(scores)

# 1: decision tree evaluated on imbalanced dataset
# evaluate pipeline
from sklearn.metrics import classification_report, accuracy_score, make_scorer

# Variables for average classification report
originalclass = []
predictedclass = []

#Make our customer score
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Nested CV with parameter optimization
scores = cross_val_score(model, X, y,scoring=make_scorer(classification_report_with_accuracy_score), cv=cv, n_jobs=-1)

# Average values in classification report for all folds in a K-fold Cross-validation
print(classification_report(originalclass, predictedclass))

exit()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y,scoring=make_scorer(classification_report_with_accuracy_score), cv=cv, n_jobs=-1)





#print(measure1, '%.3f' % mean(scores))
exit()

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
