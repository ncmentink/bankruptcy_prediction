import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Load the correct dataset for this logistic regression
data = pd.read_csv("data_final.csv")
#data = pd.read_csv("data_1.csv")

# Set independent variable
y = data['loan_status']

# Set dependent variables
# Important to leave out baseline dummies
X = data[['term_36', 'dti', 'fico_range_low', 'age', 'sub_grade_B', 'sub_grade_C', 'sub_grade_D', 'sub_grade_E',
          'sub_grade_F', 'sub_grade_G'
          # 'loan_amnt', 'int_rate', 'open_acc', 'pub_rec', 'pay_status', 'installment',
          # 'annual_inc', 'fico_range_high', 'revol_bal','revol_util',
          # 'sub_grade_A', 'term_65' Leave out because of dummy trap
          ]]

# Randomly split into test (0.75%) and train sets (0.25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Perform logistic regression
# Set max_iter higher to ensure convergence
logistic_regression = LogisticRegression(max_iter=1500)
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

# Show heatmap of confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

# Print performance measures
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 score:", metrics.f1_score(y_test, y_pred))

# Make ROC/AUC plot
y_pred_proba = logistic_regression.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
