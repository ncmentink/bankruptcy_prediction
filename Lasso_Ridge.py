from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Load data
data = pd.read_csv("data.csv")

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']


# We standardize the data, as this is necessary for Ridge and Lasso
# Fit the scaler to X
X = StandardScaler().fit_transform(X)
print(Counter(y))
# Oversample minority class to a 1:10 ratio
X, y = SMOTE(sampling_strategy=0.1).fit_resample(X, y)
print(Counter(y))
# Under sample majority class to a 2:1 ratio
X, y = RandomUnderSampler(sampling_strategy=0.5).fit_resample(X, y)
print(Counter(y))

# C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions
# With Stratified K-fold Cross validation we found that c=0.07 gives the best recall
c = 0.07

# Lasso
lasso = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=8000)
lasso.fit(X, y)
coeff = lasso.coef_[0]

print('C:', c)
print('Coefficient of each feature:', coeff)

# Summarize feature importance
for i, v in enumerate(coeff):
    print('Feature: %0d, Score: %.5f' % (i, v))

# Plot feature importance
labels = data.columns[1:]
plt.bar([x for x in range(len(coeff))], coeff, tick_label=labels)
plt.xticks(rotation=90)
plt.ylabel('Coefficient size of Logistic Lasso')
plt.show()


# Ridge
ridge = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=8000)
ridge.fit(X, y)
coeff = ridge.coef_[0]

print('C:', c)
print('Coefficient of each feature:', coeff)


# Summarize feature importance/coefficients
for i, v in enumerate(coeff):
    print('Feature: %0d, Score: %.5f' % (i, v))

# Plot feature importance/coefficients
labels = data.columns[1:]
plt.bar([x for x in range(len(coeff))], coeff, tick_label=labels)
plt.xticks(rotation=90)
plt.ylabel('Coefficient size of Logistic Ridge')
plt.show()
