from numpy import mean
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_dataset():
    data = pd.read_csv("data.csv")
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']
    return X, y


def get_pipeline():
    # Turn one off
    model = LogisticRegression(penalty="l1", C=1, solver='saga', max_iter=8000)
    # model = LogisticRegression(penalty="l2", C=1, solver='lbfgs', max_iter=8000)

    over = SMOTE(sampling_strategy=0.1, random_state=3)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=3)
    steps = [('over', over), ('under', under), ('scale', StandardScaler()),
             ('model', model)]
    pipeline = Pipeline(steps=steps)

    return pipeline


# evaluate the model using a given test condition
def evaluate_model(cv, perf_measure):
    # get the dataset
    X, y = get_dataset()

    # get the pipeline
    pipeline = get_pipeline()

    # evaluate the model
    scores = cross_val_score(pipeline, X, y, cv=cv,
                             scoring=perf_measure, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()


# define range of folds to test
folds = range(2,25)

# create to store mean, min, max for each k
means, mins, maxs = list(), list(), list()

# evaluate each value for k
for k in folds:

    # define cross validation
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    # define performance measure
    perf_measure = "recall"

    # collect mean, min and max for k
    k_mean, k_min, k_max = evaluate_model(cv, perf_measure)

    # print the performance
    print('> folds=%d, %s = %.3f (%.3f,%.3f)' % (k, perf_measure, k_mean, k_min, k_max))

    # store mean recall
    means.append(k_mean)

    # store relative min and max
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

# figure of mean with min/max error bars, for all k
plt.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
plt.xlabel("Number of folds")
plt.ylabel("Recall")

plt.show()
