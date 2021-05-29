from numpy import mean
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def get_dataset():
    data = pd.read_csv("data.csv")
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']
    return X, y


# retrieve the model to be evaluate
def get_model():
    model = LogisticRegression(penalty="l2", C=1, solver='lbfgs', max_iter=8000)
    # model = LogisticRegression(penalty="l1", C=1, solver='saga', max_iter=8000)
    return model


# evaluate the model using a given test condition
def evaluate_model(cv, measure):
    # get the dataset
    X, y = get_dataset()
    # get the model
    model = get_model()
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring=measure, cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()

# calculate the ideal test condition
#ideal, _, _ = evaluate_model(LeaveOneOut())
#print('Ideal: %.3f' % ideal)

# define folds to test
folds = range(2,32)

# record mean and min/max of each set of results
means, mins, maxs = list(), list(), list()

# evaluate each k value
for k in folds:

    # define the test condition
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    # define performance measure
    measure = "recall"

    # evaluate k value
    # the min and max gives us an idea of the dispersion
    k_mean, k_min, k_max = evaluate_model(cv, measure)

    # report performance
    print('> folds=%d, %s = %.3f (%.3f,%.3f)' % (k, measure, k_mean, k_min, k_max))

    # store mean accuracy
    means.append(k_mean)

    # store min and max relative to the mean
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

# line plot of k mean values with min/max error bars
plt.errorbar(folds, means, yerr=[mins, maxs], fmt='o')

# plot the ideal case in a separate color
#plt.plot(folds, [ideal for _ in range(len(folds))], color='r')

# show the plot
plt.show()