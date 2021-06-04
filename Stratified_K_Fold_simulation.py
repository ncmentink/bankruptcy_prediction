import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer, accuracy_score, \
    f1_score, recall_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load data
data = pd.read_csv("data_woe.csv")

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

# Optimal C
C = [0.07]

output_file = open('classification_report.txt', 'w')

for c in C:
    # Lasso (l1), Ridge (l2); turn one off with # if desired
    # Increase no. of iterations to ensure convergence
    Models = [LogisticRegression(penalty="l1", C=c, solver='saga', max_iter=8000),
              LogisticRegression(penalty="l2", C=c, solver='lbfgs', max_iter=8000)]

    for model in Models:

        # Create lists for the average classification report
        true_labels = []
        predicted_labels = []

        # Over sample to a 1:10 ratio
        over = SMOTE(sampling_strategy=0.1, random_state=3)

<<<<<<< HEAD:Stratified_K_Fold.py
        # Under sample to a 1:2 ratio.
        # alpha = (# in minority class / # in majority class after resampling)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=3)

        # Folds are stratified; therefore have the same ratio as original dataset.
        # Original: 1:32 ratio (3% 1-class, 97% 0-class);
        # After pipeline 1:2 ratio (33% 1-class, 67% 0-class).
        steps = [('over', over), ('under', under), ('scale', StandardScaler()),
                 ('model', model)]
        pipeline = Pipeline(steps=steps)

        # Nested CV with parameter optimization
        # Pick optimization measure function: accuracy, recall, f1-score, roc_auc
        cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True,
                                                           random_state=0),
                        scoring=make_scorer(classification_report_with_recall))

        # Get the average classification report of all K estimations
        print('C:', c)
        print('model:', model.penalty)
        print("ROC_AUC score:", roc_auc_score(true_labels, predicted_labels))
        print(classification_report(true_labels, predicted_labels))

        output_file.write("C = %s, model =  %s \n" % (c, model.penalty))
        output_file.write("ROC_AUC score: %s \n" % roc_auc_score(true_labels, predicted_labels))
        output_file.write("%s\n" % classification_report(true_labels, predicted_labels))

output_file.close()
=======
print(np.mean(recall0))
print(np.var(recall0))
>>>>>>> 236758d8cfef8d6bf7f6bc830a45b438405d5c3a:Stratified_K_Fold_simulation.py
