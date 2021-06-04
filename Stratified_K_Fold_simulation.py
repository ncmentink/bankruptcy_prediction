import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer, accuracy_score, \
    f1_score, recall_score, roc_curve, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


import numpy as np

variance = [65,68,68,68,68,68,68,69,69,69,69,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,71,71,71,71,71,71,71,71,71,71,71,72,72,72,72,72,73,73,73,73]
print(len(variance))
print(np.mean(variance))
print(np.var(variance))

# Load data
data = pd.read_csv("data.csv")

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

def classification_report_with_f1(y_true, y_pred):
    true_labels.extend(y_true)
    predicted_labels.extend(y_pred)

    return f1_score(y_true, y_pred)


def classification_report_with_recall(y_true, y_pred):
    true_labels.extend(y_true)
    predicted_labels.extend(y_pred)

    return recall_score(y_true, y_pred)


def classification_report_with_accuracy(y_true, y_pred):
    true_labels.extend(y_true)
    predicted_labels.extend(y_pred)

    return accuracy_score(y_true, y_pred)


def classification_report_with_roc_auc(y_true, y_pred):
    true_labels.extend(y_true)
    predicted_labels.extend(y_pred)

    return roc_auc_score(y_true, y_pred)


# Choose a range of values for C = 1/Lambda.
# By decreasing C, we increase sparsity and get more zero predictions.
# C = [10, 5, 1, 0.5, 0.1, 0.05, 0.001]
C = [0.07]

output_file = open('classification_report.txt', 'w')


recall0 = []
for i in range(50):

    for c in C:

        # First model: Lasso (l1), second: Ridge (l2)
        # Increase no. of iterations to ensure convergence
        # Turn one off with # if desired
        # Models = [LogisticRegression(penalty="l1", C=c, solver='saga', max_iter=8000),
        #           LogisticRegression(penalty="l2", C=c, solver='lbfgs', max_iter=8000)]
        Models = [LogisticRegression(penalty="l1", C=c, solver='saga', max_iter=8000)]

        for model in Models:

            # Create lists for the average classification report
            true_labels = []
            predicted_labels = []

            # Over sample to a 1:10 ratio
            over = SMOTE(sampling_strategy=0.1, random_state=3)

            # Under sample to a 1:2 ratio.
            # alpha = (# in minority class / # in majority class after resampling)
            under = RandomUnderSampler(sampling_strategy=0.5, random_state=3)

            # Folds are stratified; therefore have the same ratio as original dataset.
            # Original: 1:32 ratio (3% 1-class, 97 0-class);
            # After pipeline 1:2 (33% 1-class, 66% 0-class).
            steps = [('over', over), ('under', under), ('scale', StandardScaler()),
                     ('model', model)]
            pipeline = Pipeline(steps=steps)

            # Nested CV with parameter optimization
            # Pick optimization measure: accuracy, recall, f1-score
            cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True),
                            scoring=make_scorer(classification_report_with_recall))

            # Get average classification report of all K estimations
            print('C:', c)
            print('model:', model.penalty)
            print("ROC_AUC score:", roc_auc_score(true_labels, predicted_labels))
            print(classification_report(true_labels, predicted_labels))
            recall0.append(classification_report_with_recall(true_labels,predicted_labels))
            output_file.write("C = %s, model =  %s \n" % (c, model.penalty))
            output_file.write("ROC_AUC score: %s \n" % roc_auc_score(true_labels, predicted_labels))
            output_file.write("%s\n" % classification_report(true_labels, predicted_labels))

            # Make ROC/AUC plot
            fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
            auc = roc_auc_score(true_labels, predicted_labels)
            plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
            plt.legend(loc=4)
            # plt.show()

output_file.close()

print(np.mean(recall0))
print(np.var(recall0))
