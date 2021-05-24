import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer, accuracy_score, f1_score, recall_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load data
data = pd.read_csv("data.csv")

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']


def classification_report_with_f1_score(y_true, y_pred):
    true_labels.extend(y_true)
    predicted_labels.extend(y_pred)

    return f1_score(y_true, y_pred)


# C = 1/Lambda. By decreasing C, we increase sparsity and hence should get more zero predictions.
C = [10, 5, 1, 0.5, .1, 0.05, .001]

for c in C:
    # Pick a model
    # Lasso
    model = LogisticRegression(penalty="l1", solver='saga', max_iter=8000)
    # Ridge
    # model = LogisticRegression(penalty="l2", solver='lbfgs', max_iter=8000)

    # Variables for average classification report
    true_labels = []
    predicted_labels = []

    # Over sample to a 1:10 ration
    over = SMOTE(sampling_strategy=0.1)

    # Under sample to a 1:2 ratio
    under = RandomUnderSampler(sampling_strategy=0.5)

    # Pipelines help avoid leaking statistics from your test data into the trained model in cross-validation,
    # by ensuring that the same samples are used to train the transformers and predictors. Also, the scaler is fit on
    # the training data, transforms the train data, models are fitted on the train data, and the scaler is used to
    # transform the test data. Therefore, the test data is not used to determine the scaling parameters.
    #
    # After under and over sampling we standardize the data, as this is necessary for Ridge and Lasso. If we standardize
    # instead of normalize, we still keep the interpretability of our coefficients.
    # For the Lasso, the regularization penalty is comprised of the sum of the absolute value of the coefficients,
    # therefore we need to standardize the data so the coefficients are all based on the same scale.
    #
    # The folds of the cross-validation are stratified, which means they will have the same class distribution as the
    # original dataset, in this case a 1:32 ratio (3% 1-class, 97 0-class) and later 1:2 (33% 1-class, 66% 0-class).

    steps = [('over', over), ('under', under), ('scale', StandardScaler()), ('model', model)]
    pipeline = Pipeline(steps=steps)

    # Nested CV with parameter optimization (f1_score)
    cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
                    scoring=make_scorer(classification_report_with_f1_score))

    # Average values in classification report for all folds in a K-fold Cross-validation
    print('C:', c)
    print('model:', model.penalty)
    print(classification_report(true_labels, predicted_labels))
