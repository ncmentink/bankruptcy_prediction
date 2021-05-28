import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv("data.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Data exploration: descriptive statistics
pd.set_option('display.expand_frame_repr', False)
#print(data.describe(include="all"))

# x = data.iloc[:,[1, 13, 16, 19, 27, 29, 32, 34, 35, 36, 37, 38, 43, 44, 45, 46, 48, 49, 50, 56, 57, 59, 64, 72, 74, 75, 76, 77, 80, 85, 86]]
# print(x)

#print(data.iloc[[420,69,6000,1234,3],[1, 13, 16, 19, 27, 29, 32, 34, 35, 36, 37, 38, 43, 44, 45, 46, 48, 49, 50, 56, 57, 59, 64, 72, 74, 75, 76, 77, 80, 85, 86]])



# # Plotting Boxplots of the numerical features, first plot is of first 48 features
# plt.figure(figsize = (20,20))
# ax =sns.boxplot(data = data.iloc[:,:48], orient="h")
# ax.set_title('Boxplot bank data (first 47 features)', fontsize = 18)
# ax.set(xscale="log")
# plt.show()
#
# # Second plot is of last 48 features
# plt.figure(figsize = (20,20))
# ax =sns.boxplot(data = data.iloc[:,48:], orient="h")
# ax.set_title('Boxplot bank data (last 48 features)', fontsize = 18)
# ax.set(xscale="log")
# plt.show()



# for i in range(len(data)):
#     if data.iloc[i,47] > 1000:
#         print(data.iloc[i,47])
#         print('Bankruptcy: ', data.iloc[i,0])
#     else:
#         continue
# exit()

# Missing values: none
count_NA = data.isna().sum()
# print(count_NA)


# Check for duplicates: none
# print(data.duplicated().sum())


# Calculate percentage of bankruptcies: only 3%!
# Very imbalanced class labels
count_defaults = data["Bankrupt?"].value_counts().to_dict()
# print(count_defaults[1]/(count_defaults[0] + count_defaults[1]))
sns.countplot(x=data['Bankrupt?'])
plt.show()


# Data transformation
# 1) Resample by means of SMOTE since only 3% of 1-class
# 2) Scale the data

# 1) Resample by means of SMOTE, to 50/50
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']
X_smote, y_smote = SMOTE().fit_resample(X, y)

# 2) Create scaled data
# Makes a large difference in performance
data = y_smote.to_frame().join(X_smote)
count = 0
not_scaled = []
for col in data.columns:
    if max(data[col]) > 1:
        # print("Unscaled : ", col)
        count += 1
        not_scaled.append(col)

scaling_function = MinMaxScaler()
data[not_scaled] = scaling_function.fit_transform(data[not_scaled])

X_smote_sc = data.drop('Bankrupt?', axis=1)
y_smote_sc = data['Bankrupt?']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# We standardize the data, as this is necessary for Ridge and Lasso
# For the Lasso, the regularization penalty is comprised of the sum of the absolute value of the coefficients,
# therefore we need to standardize the data so the coefficients are all based on the same scale.
sc = StandardScaler()

# Fit the scaler to the training data and transform
X_train_std = sc.fit_transform(X_train)

# Apply the scaler to the test data
X_test_std = sc.transform(X_test)

# C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions
C = [0.07]
#C = [10, 5, 1, 0.5, .1, 0.05, .001]


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
model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000)

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
