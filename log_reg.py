import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load the variables for logistic regression
# Load data
data = pd.read_csv("data.csv")

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']
data_woe = pd.read_csv("data_woe.csv")
X_woe = data_woe.drop('Bankrupt?', axis=1)
y_woe = data_woe['Bankrupt?']

c = [0.07]

# Pick a model
# Lasso
model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000, C=c)
# Ridge
# model = LogisticRegression(penalty="l2", solver='lbfgs', max_iter=8000, C=c)

# 1) LOGISTIC REGRESSION WITHOUT RESAMPLING, WITHOUT SCALING
# Randomly split into train (0.75%) and test sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Perform logistic regression
# Set max_iter higher to ensure convergence
lr = LogisticRegression(max_iter=1500)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Show heatmap of confusion matrix
confusionmatrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusionmatrix, annot=True)
plt.show()

# Print performance measures
print("lr without resampling, unscaled")
print(classification_report(y_test, y_pred))


exit()

# Make ROC/AUC plot
y_pred_proba = lr.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# 2) LOGISTIC REGRESSION WITH RESAMPLING, WITHOUT SCALING
# Randomly split into test (0.75%) and train sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.25, random_state=0)

# Perform logistic regression
# Set max_iter higher to ensure convergence
lr_smote = LogisticRegression(max_iter=1500)
lr_smote.fit(X_train, y_train)
y_smote_pred = lr_smote.predict(X_test)

# Show heatmap of confusion matrix
confusionmatrix = pd.crosstab(y_test, y_smote_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusionmatrix, annot=True)
plt.show()

# Print performance measures
print("lr with resampling, unscaled")
print(classification_report(y_test, y_smote_pred))

# Make ROC/AUC plot
y_pred_proba = lr_smote.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# 3) LOGISTIC REGRESSION WITH RESAMPLING, WITH SCALING
# Randomly split into test (0.75%) and train sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_smote_sc, y_smote_sc, test_size=0.25, random_state=0)

# Perform logistic regression
# Set max_iter higher to ensure convergence
lr_smote_sc = LogisticRegression(max_iter=1500)
lr_smote_sc.fit(X_train, y_train)
y_smote_sc_pred = lr_smote_sc.predict(X_test)

# Show heatmap of confusion matrix
confusionmatrix = pd.crosstab(y_test, y_smote_sc_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusionmatrix, annot=True)
plt.show()

# Print performance measures
print("lr with resampling, with scaling")
print(classification_report(y_test, y_smote_sc_pred))

# Make ROC/AUC plot
y_pred_proba = lr_smote_sc.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# 4) LOGISTIC REGRESSION WITH WOE (UNSCALED, WITH RESAMPLING)
# Randomly split into test (0.75%) and train sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_woe, y_woe, test_size=0.25, random_state=0)

# Perform logistic regression
# Set max_iter higher to ensure convergence
lr_woe = LogisticRegression(max_iter=1500)
lr_woe.fit(X_train, y_train)
y_woe_pred = lr_woe.predict(X_test)

# Show heatmap of confusion matrix
confusionmatrix = pd.crosstab(y_test, y_woe_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusionmatrix, annot=True)
plt.show()

# Print performance measures
print("lr with WOE (with resampling, unscaled)")
print(classification_report(y_test, y_woe_pred))

# Make ROC/AUC plot
y_pred_proba = lr_woe.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
