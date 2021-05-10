import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


# Load data
data = pd.read_csv("data.csv")

# Data exploration: descriptive statistics
pd.set_option('display.expand_frame_repr', False)
print(data.describe(include="all"))


# Missing values
# No missing values
count_NA = data.isna().sum()
# print(count_NA)


# Count frequencies (to detect anomalies)
# for column in data:
#    counts = data[column].value_counts().to_dict()
#    print(counts)


# Data transformation
# 1) Resample by means of SMOTE: oversample y=1 up to a 50/50 ratio
# 2) Scale the data

# 1) Calculate percentage of bankruptcies: only 3%!
count_defaults = data["Bankrupt?"].value_counts().to_dict()
# print(count_defaults[1]/(count_defaults[0] + count_defaults[1]))

# Therefore resample by means of SMOTE
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']
X_smote, y_smote = SMOTE().fit_resample(X, y)


# 2) Create scaled data
# Makes a HUGE difference in performance
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


# VARIABLE SELECTION
# 1) Exclude variables with multicollinearity
# 2) Check if correlated with Y
# 3) Check if enough variability
# 4) (In case of WOE/IV) IV criteria


def plot_confusion_matrix(cm, title='Confusion matrix', labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
    plt.show()


# 1) Get correlation plot
# plot_confusion_matrix(data.corr())


# Histograms
# Plot them one by one to see details
# for column in data:
#    data[[column]].hist(bins=50)
#    plt.show()

# Plot per 6
# for subset in range(0, 102, 6):
#    data.iloc[:, subset:subset+9].hist(figsize=(40, 30), bins=50)
#    plt.show()

"""
pd.crosstab(data["Bankrupt?"], data[" ROA(C) before interest and depreciation before interest"], normalize="index").plot(kind='bar')
plt.title('Default frequency for New worth/Assets')
plt.xlabel('Sub_grade')
plt.ylabel('Frequency')
plt.show()
"""