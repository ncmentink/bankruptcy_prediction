from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the variables for logistic regression
from data_prep import X, y, X_smote, y_smote, X_smote_sc, y_smote_sc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# First we scale the data, as this is necessary for Ridge and Lasso
# For the Lasso, the regularization penalty is comprised of the sum of the absolute value of the coefficients,
# therefore we need to scale the data so the coefficients are all based on the same scale.
sc = StandardScaler()

# Fit the scaler to the training data and transform
X_train_std = sc.fit_transform(X_train)

# Apply the scaler to the test data
X_test_std = sc.transform(X_test)

# C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions
C = [10, 5, 1, 0.5, .1, 0.05, .001]


# Lasso
for c in C:
    clf = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=8000)
    clf.fit(X_train_std, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train_std, y_train))
    print('Test accuracy:', clf.score(X_test_std, y_test))
    print('')

# Ridge
for c in C:
    clf = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=8000)
    clf.fit(X_train_std, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train_std, y_train))
    print('Test accuracy:', clf.score(X_test_std, y_test))
    print('')





"""
X_train, X_test, y_train, y_test = train_test_split(X_smote_sc, y_smote_sc, test_size=0.25, random_state=0)

# Logistic Lasso 
log_las = LogisticRegression(penalty='l1', solver='saga', max_iter=8000, C=10)
log_las.fit(X_train, y_train)
y_pred = log_las.predict(X_test)
print(log_las.coef_)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Logistic Ridge
log_rid = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=8000, C=10)
log_rid.fit(X_train, y_train)
y_pred = log_rid.predict(X_test)
print(log_rid.coef_)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
"""