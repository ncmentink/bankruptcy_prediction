import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer, accuracy_score, \
    f1_score, recall_score, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import json

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


# C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions.
# C = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05,
#      0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4,
#      1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3, 4, 5, 6, 7, 8, 9, 10]
C = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
     0.18, 0.19, 0.20, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33,
     0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
# C = [10, 5 , 1, 0.5, 0.1, 0.05, 0.001]

output_file = open('classification_report.txt', 'w')

X_plot = []
y_plot = []
precision0 = []
precision1 = []
recall0 = []
recall1 = []
f1_score0 = []
f1_score1 = []
ROC_AUC = []
accuracy = []

for c in C:
    # First model: Lasso (slow), second: Ridge (quick)
    # Turn one off with # if desired
    # Models = [LogisticRegression(penalty="l1", C=c, solver='saga', max_iter=8000)
    #     , LogisticRegression(penalty="l2", C=c, solver='lbfgs', max_iter=8000)]
    Models = [LogisticRegression(penalty="l2", C=c, solver='lbfgs', max_iter=8000)]
    #Models = [LogisticRegression(penalty="l1", C=c, solver='saga', max_iter=8000)]
    for model in Models:

        # Create lists for the average classification report
        true_labels = []
        predicted_labels = []

        # Over sample to a 1:10 ratio
        over = SMOTE(sampling_strategy=0.1)

        # Under sample to a 1:2 ratio
        under = RandomUnderSampler(sampling_strategy=0.5)

        # Pipelines help avoid leaking statistics from your test data into the trained model in cross-validation,
        # by ensuring that the same samples are used to train the transformers and predictors. Also, the scaler is fit
        # on the training data, transforms the train data, models are fitted on the train data, and the scaler is used
        # to transform the test data. Therefore, the test data is not used to determine the scaling parameters.
        #
        # After under and over sampling we standardize the data, as this is necessary for Ridge and Lasso. If we
        # standardize instead of normalize, we still keep the interpretability of our coefficients.
        # For the Lasso, the regularization penalty is comprised of the sum of the absolute value of the coefficients,
        # therefore we need to standardize the data so the coefficients are all based on the same scale.
        #
        # The folds of the cross-validation are stratified, which means they have the same class distribution as the
        # original dataset, in this case a 1:32 ratio (3% 1-class, 97 0-class) and later 1:2 (33% 1-class, 66% 0-class).

        steps = [('over', over), ('under', under), ('scale', StandardScaler()), ('model', model)]
        pipeline = Pipeline(steps=steps)

        # Nested CV with parameter optimization
        # Set measure to optimize by changing scoring function: accuracy, recall or f1-score
        cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
                        scoring=make_scorer(classification_report_with_recall))

        # Average values in classification report for all folds in a Stratified K-fold Cross-validation
        print('C:', c)
        print('model:', model.penalty)
        print("ROC_AUC score:", roc_auc_score(true_labels, predicted_labels))
        print(classification_report(true_labels, predicted_labels))

        output_file.write("C = %s, model =  %s \n" % (c, model.penalty))
        output_file.write("ROC_AUC score: %s \n" % roc_auc_score(true_labels, predicted_labels))
        output_file.write("%s\n" % classification_report(true_labels, predicted_labels))

        # Make ROC/AUC plot
        # fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        auc = roc_auc_score(true_labels, predicted_labels)
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # plt.legend(loc=4)
        # plt.show()

        precision, recall, fscore, support = score(true_labels, predicted_labels, average=None)

        precision0.append(precision[0])
        precision1.append(precision[1])
        recall0.append(recall[0])
        recall1.append(recall[1])
        f1_score0.append(fscore[0])
        f1_score1.append(fscore[1])

        ROC_AUC.append(roc_auc_score(true_labels, predicted_labels))
        accuracy.append(accuracy_score(true_labels, predicted_labels, normalize=True))

    X_plot.append(c)
    y_plot = recall1

print(precision0)
print(precision1)
print(recall0)
print(recall1)
print(f1_score0)
print(f1_score1)
print(ROC_AUC)
print(accuracy)


fig = plt.figure()
ax = plt.axes()
plt.title("ROC_AUC score for different values of C")
plt.xlabel('Different values of C', fontsize=18)
plt.ylabel('ROC_AUC score', fontsize=16)
ax.plot(X_plot, y_plot, marker='o')
plt.show()

with open("lists for different C's/C.txt", 'w') as f:
    f.write(json.dumps(C))
with open("lists for different C's/L1_precision1.txt", 'w') as f:
    f.write(json.dumps(precision0))
with open("lists for different C's/L1_precision2.txt", 'w') as f:
    f.write(json.dumps(precision1))
with open("lists for different C's/L1_recall1.txt", 'w') as f:
    f.write(json.dumps(recall0))
with open("lists for different C's/L1_recall2.txt", 'w') as f:
    f.write(json.dumps(recall1))
with open("lists for different C's/L1_f1_score1.txt", 'w') as f:
    f.write(json.dumps(f1_score0))
with open("lists for different C's/L1_f1_score2.txt", 'w') as f:
    f.write(json.dumps(f1_score1))
with open("lists for different C's/L1_ROC_AUC.txt", 'w') as f:
    f.write(json.dumps(ROC_AUC))
with open("lists for different C's/L1_accuracy.txt", 'w') as f:
    f.write(json.dumps(accuracy))




output_file.close()
