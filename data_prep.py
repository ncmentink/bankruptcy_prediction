import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv("data.csv")

# Data exploration: descriptive statistics
pd.set_option('display.expand_frame_repr', False)
print(data.describe(include="all"))


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
