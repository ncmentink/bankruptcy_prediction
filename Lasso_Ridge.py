import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load the variables for logistic regression
from data_prep import X, y, X_smote, y_smote, X_smote_sc, y_smote_sc


# Logistic Lasso
# C = 1/Lambda. By increasing C, we decrease sparsity and hence should get more nonzero predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

log_las = LogisticRegression(penalty='l1', solver='saga', max_iter=8000, C=10000)
log_las.fit(X_train, y_train)
y_pred = log_las.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Logistic Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

log_rid = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=8000, C=10000)
log_rid.fit(X_train, y_train)
y_pred = log_rid.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
