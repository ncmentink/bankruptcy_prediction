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
print(data.describe(include="all"))

# Plotting Boxplots of the numerical features, first plot is of first 48 features
plt.figure(figsize = (20,20))
ax =sns.boxplot(data = data.iloc[:,:48], orient="h")
ax.set_title('Boxplot bank data (first 47 features)', fontsize = 18)
ax.set(xscale="log")
plt.show()

# Second plot is of last 48 features
plt.figure(figsize = (20,20))
ax =sns.boxplot(data = data.iloc[:,48:], orient="h")
ax.set_title('Boxplot bank data (last 48 features)', fontsize = 18)
ax.set(xscale="log")
plt.show()


exit()


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

# # Ridge
# for c in C:
#     clf = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=8000)
#     clf.fit(X_train_std, y_train)
#     print('C:', c)
#     print('Coefficient of each feature:', clf.coef_)
#     print('Training accuracy:', clf.score(X_train_std, y_train))
#     print('Test accuracy:', clf.score(X_test_std, y_test))
#     print('')