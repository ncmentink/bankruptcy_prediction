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


# Create boxplots
#Plotting Boxplots of the numerical features, first plot is of first 48 features
plt.figure(figsize = (20,20))
ax =sns.boxplot(data = data.iloc[:,:48], orient="h")
ax.set_title('Boxplot bank data (first 47 features)', fontsize = 18)
ax.set(xscale="log")
plt.show()

# Second plot is of last 48 features
plt.figure(figsize = (20,20))
ax =sns.boxplot(data = data.iloc[:,48:], orient="h")
ax.set_title('Boxplot bank data (last 48 features)', fontsize = 18)
ax.set(xscale="log")
plt.show()
