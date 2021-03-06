import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load the data
data = pd.read_csv("data.csv")
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# Load the WOE binned data
data_woe = pd.read_csv("data_woe.csv")
X_woe = data_woe.drop('Bankrupt?', axis=1)
y_woe = data_woe['Bankrupt?']

# Pick a model
# Logistic Regression with Lasso or Ridge
model = LogisticRegression(solver='lfbgs', penalty="l1", max_iter = 8000)
# model = LogisticRegression(solver='saga', penalty="l2", max_iter = 8000)

# 1) LOGISTIC LASSO/RIDGE REGRESSION WITHOUT OVER/UNDER SAMPLING
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
print(metrics.classification_report(y_test, y_pred))

# Make ROC/AUC plot
y_pred_proba = lr.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# 2) LOGISTIC REGRESSION WITH WOE BINNED DATA, WITHOUT OVER/UNDER SAMPLING
# Randomly split into test (0.75%) and train sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_woe, y_woe, test_size=0.25, random_state=0, stratify=y_woe)

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
print(metrics.classification_report(y_test, y_woe_pred))

# Make ROC/AUC plot
y_pred_proba = lr_woe.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
