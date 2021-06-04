from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# By making this loop, we can investigate the outcomes of several random states
# However we chose j = 3 since that seemed to pick an average amount of features
for j in range(3, 4):
    print(j)

    # Load data
    data = pd.read_csv("data.csv")

    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']


    # We standardize the data, as this is necessary for Ridge and Lasso
    # Fit the scaler to X
    X = StandardScaler().fit_transform(X)
    print(Counter(y))
    # Oversample minority class to a 1:10 ratio
    X, y = SMOTE(sampling_strategy=0.1, random_state=j).fit_resample(X, y)
    print(Counter(y))
    # Under sample majority class to a 1:2 ratio
    X, y = RandomUnderSampler(sampling_strategy=0.5, random_state=j).fit_resample(X, y)
    print(Counter(y))

    # C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions
    # With Stratified K-fold Cross validation we found that c=0.07 gives the best recall
    c = 0.07

    # Lasso
    print("Lasso")
    lasso = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=8000)
    lasso.fit(X, y)
    coeff = lasso.coef_[0]
    print('C:', c)
    print('Coefficient of each feature:', coeff)

    # Summarize feature importance
    coeff_index = []
    nonzero_coeff = []
    for i, v in enumerate(coeff):
        print('Feature: %0d, Score: %.5f' % (i+1, v))
        if v != 0:
            coeff_index.append(i+1)
            nonzero_coeff.append(v)
    print(coeff_index)
    print(len(coeff_index))
    print(nonzero_coeff)


    # Plot feature importance
    labels = data.columns[coeff_index]
    plt.bar([x for x in range(len(nonzero_coeff))], nonzero_coeff, tick_label=labels)
    plt.xticks(rotation=90)
    plt.ylabel('Coefficient size of Logistic Lasso')
    xlocs = [i + 1 for i in range(0, 31)]
    xlabs = [i / 2 for i in range(0, 31)]
    # Alter placements of coefficients
    alter = [-1.5, -1.5, -1.5, -1.5, 1, 1, 1, -1.5, -1.5, 1, 1, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1, -1.5, 1, -1.5, 1, 1, -1.5, -1.5, 1, 1, 1, -1.5, -1.5, -1.5]
    print(alter[1])
    print(len(alter))
    for i, v in enumerate(nonzero_coeff):
        plt.text(xlocs[i] - 1.47, v + 0.02*alter[i],  '%.2f'% v)
    plt.show()


    # Ridge
    print("Ridge")
    ridge = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=8000)
    ridge.fit(X, y)
    coeff = ridge.coef_[0]
    print('C:', c)
    print('Coefficient of each feature:', coeff)


    # Summarize feature importance/coefficients
    for i, v in enumerate(coeff):
        print('Feature: %0d, Score: %.5f' % (i+1, v))
