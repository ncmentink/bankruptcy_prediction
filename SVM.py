import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold


# Load the correct dataset for this logistic regression
data = pd.read_csv("data.csv")

print(data)
exit()

# Set independent variable
y = data["Bankrupt?"]

X = data.drop(['Bankrupt?'],axis=1)


X_smote, y_smote = SMOTE().fit_resample(X, y)


# Randomly split into test (0.75%) and train sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.25, random_state=0)


# # Set RandomForestClassifier as estimator for RFECV
# cart = RandomForestClassifier(random_state=42)
# # Minimum number of features to consider
# min_features_to_select = 1
# # Set number of folds
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
# # Set cross-validation process
# rfecv = RFECV(estimator=cart, step=1, cv=cv,
#               scoring='accuracy',
#               min_features_to_select=min_features_to_select, n_jobs=1)
# # Fit the model
# rfecv.fit(X_train, y_train)
#
# print("Optimal number of features : %d" % rfecv.n_features_)
#
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(min_features_to_select,
#                len(rfecv.grid_scores_) + min_features_to_select),
#          rfecv.grid_scores_)
# plt.show()


# var_thres = VarianceThreshold(2.0)
# var_thres.fit(X)
# required_features = [col for col in X.columns if col in X.columns[var_thres.get_support()]]
# print(required_features)
#
# df_v1 = data[required_features]
# df_v1.head()
#
# #Checking for multicollinearity
# df_v1_corr = df_v1.corr()
# df_v1_corr.style.background_gradient(cmap='coolwarm')

# Set dependent variables
# Important to leave out baseline dummies to prevent dummy trap
# X = data[[' Operating Expense Rate',
#  ' Research and development expense rate',
#  ' Interest-bearing debt interest rate',
#  ' Revenue Per Share (Yuan ¥)',
#  ' Total Asset Growth Rate',
#  ' Net Value Growth Rate',
#  ' Current Ratio',
#  ' Quick Ratio',
#  ' Total debt/Total net worth',
#  ' Accounts Receivable Turnover',
#  ' Average Collection Days',
#  ' Inventory Turnover Rate (times)',
#  ' Fixed Assets Turnover Frequency',
#  ' Revenue per person',
#  ' Allocation rate per person',
#  ' Quick Assets/Current Liability',
#  ' Cash/Current Liability',
#  ' Inventory/Current Liability',
#  ' Long-term Liability to Current Assets',
#  ' Current Asset Turnover Rate',
#  ' Quick Asset Turnover Rate',
#  ' Cash Turnover Rate',
#  ' Fixed Assets to Assets',
#  ' Total assets to GNP price']]




svc = SVC(C=1, kernel='linear', gamma=0.001,probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print('Times predicted bankrupt:', sum(y_pred))


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

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

