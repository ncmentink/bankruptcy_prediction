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

from data_prep import y_smote_sc, X_smote_sc, y_smote, X_smote

# Load the correct dataset for this logistic regression
data = y_smote_sc.to_frame().join(X_smote_sc)

# Randomly split into train (0.75%) and test sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_smote_sc, y_smote_sc, test_size=0.25, random_state=0)

lr = LogisticRegression(max_iter=1500)
rfecv = RFECV(estimator=lr, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_smote_sc, y_smote_sc)

plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

most_relevant_cols = data.iloc[:, 1:].columns[np.where(rfecv.support_ == True)]
print("Most relevant features are: ")
print(most_relevant_cols)

# Calculate accuracy scores
X_new = data[most_relevant_cols]
initial_score = cross_val_score(lr, X_smote_sc, y_smote_sc, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))
fe_score = cross_val_score(lr, X_new, y_smote_sc, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))


"""
#RFECV Tobi
# Set RandomForestClassifier as estimator for RFECV
cart = RandomForestClassifier(random_state=42)
# Minimum number of features to consider
min_features_to_select = 1
# Set number of folds
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
# Set cross-validation process
rfecv = RFECV(estimator=cart, step=1, cv=cv,
              scoring='accuracy',
              min_features_to_select=min_features_to_select, n_jobs=1)
# Fit the model
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()

most_relevant_cols = data.iloc[:, 1:].columns[np.where(rfecv.support_ == True)]
print("Most relevant features are: ")
print(most_relevant_cols)
"""



# Randomly split NEW DATA into train (0.75%) and test sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_smote_sc, test_size=0.25, random_state=0)


svc = SVC(C=1, kernel='linear', gamma=0.001,probability=True)
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

