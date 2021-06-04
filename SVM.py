import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold

from data_prep import X, y


# Create the correct dataset for this logistic regression
data = y.to_frame().join(X)


# Randomly split into train (0.75%) and test sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# RFECV Tobi
# Set RandomForestClassifier as estimator for RFECV
# estimator =RandomForestClassifier(random_state=42)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

lr = LogisticRegression(max_iter=10000, solver="saga")
lr.fit(X_train, y_train)
rfecv = RFECV(estimator=lr, step=1, cv=StratifiedKFold(2), scoring='accuracy', min_features_to_select=1, n_jobs=1)
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

most_relevant_cols = data.iloc[:, 1:].columns[np.where(rfecv.support_==True)]
print("Most relevant features are: ")
print(most_relevant_cols)

# Calculate accuracy scores
X_new = data[most_relevant_cols]
initial_score = cross_val_score(lr, X, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))
fe_score = cross_val_score(lr, X_new, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))

"""
# WAY NR 2 OF CHECKING IMPORTANCE
# get importance
importance = lr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
"""


# PERFORM SVC WITH FEATURES BASED ON STRATIFIED K FOLD CROSS VALIDATION FOR LOGISTIC REGRESSION

# Randomly split NEW DATA into train (0.75%) and test sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=0)


svc = SVC(C=1, kernel='linear', gamma=0.001, probability=True, n_jobs=-1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print('Times predicted bankrupt:', sum(y_pred))


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

print(classification_report(y_test, y_pred))
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 score:", metrics.f1_score(y_test, y_pred))


y_pred_proba = svc.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
