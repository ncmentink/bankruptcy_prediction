import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Load data
data = pd.read_csv("data.csv")

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']


# FEATURE SELECTION: For a small number of features, this can be done by hand
# 1) Exclude variables with (too high) multicollinearity
#       1.1: Look at correlation
#       1.2: VIF value: cutoff of 2.5
# 2) Check if correlated with Y / Check which features contribute most to prediction (accuracy)
#       2.1: Recursive Feature Elimination Cross Validation
#       2.2: Coefficients in Log Reg
# 3) Bin features on their Weight-Of-Evidence value: IV criteria
#       See woe_iv.py


# 1.1 Get correlation plot
def plot_confusion_matrix(cm, title='Confusion matrix', labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
    plt.show()

# plot_confusion_matrix(data.corr())


# 1.2 Use vif dataframe to show initial multicollinearity of the dataset
vif_data = pd.DataFrame()
vif_data['Features'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

# Remove highly related variables
while any(vif_data['VIF'] >= 2.5):
    i = vif_data['VIF'].idxmax()
    X.drop(X.columns[i], axis=1, inplace=True)
    X.reset_index(drop=True)
    vif_data = pd.DataFrame()
    vif_data['Features'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
print(len(vif_data))


# 2.1 Recursive Feature Elimination Cross Validation for Logistic Regression

# Randomly split into train (0.75%) and test sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the scaler to the training data and transform
X_train = StandardScaler().fit_transform(X_train)

# Apply the scaler to the test data
X_test = StandardScaler().transform(X_test)

lr = LogisticRegression(max_iter=1500)
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

most_relevant_cols = data.iloc[:, 1:].columns[np.where(rfecv.support_ == True)]
print("Most relevant features are: ")
print(most_relevant_cols)

# Calculate accuracy scores
X_new = data[most_relevant_cols]
initial_score = cross_val_score(lr, X, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))
fe_score = cross_val_score(lr, X_new, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))


# 2.2 Check importance by means of coefficient size of Logistic Regression
importance = lr.coef_[0]

# Summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

# Plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
