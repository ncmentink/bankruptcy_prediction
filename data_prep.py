import pandas as pd
import matplotlib.pyplot as plt


# Load data
data = pd.read_csv("data.csv")


# Data exploration: descriptive statistics
pd.set_option('display.expand_frame_repr', False)
print(data.describe(include="all"))


# Missing values
# No missing values
count_NA = data.isna().sum()
print(count_NA)


# Count frequencies (to detect anomalies)
# for column in data:
#    counts = data[column].value_counts().to_dict()
#    print(counts)


# Calculate percentage of defaults in dataset
# Only 3% has gone bankrupt
count_defaults = data["Bankrupt?"].value_counts().to_dict()
print(count_defaults[1]/(count_defaults[0] + count_defaults[1]))


# Data transformation
# TO DO:
# VARIABLE NAMES AANPASSEN
# VARIABLE TYPE: IS ALLES CONTINUOUS? OOK CATEGORICAL?
# ZOEKEN NAAR TE HOGE CORRELATIES TUSSEN VARIABELEN: MULTICOLLINEARTIY
# TESTEN VOOR CORRELATIE MET DE Y
# IS ER GENOEG VARIABILITY IN EEN VARIABLE?
# (wanneer alles in 1 waarde valt is een variabele niet informatief!)


# One-hot encoding for categorical variables
# Nog aanpassen
"""
cat_vars = ['sub_grade', 'term']
for var in cat_vars:
    # cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data_join = data.join(cat_list)
    data = data_join

cat_vars = ['sub_grade', 'term']
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data[to_keep]
"""


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


# Get correlation plot
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

pd.crosstab(data["Bankrupt?"], data[" Net worth/Assets"], normalize=False).plot(kind='bar')
plt.title('Default frequency for New worth/Assets')
plt.xlabel('Sub_grade')
plt.ylabel('Frequency')
plt.show()


""""
# Write data to csv after transformations
data_final.to_csv('data_final.csv', index=False)
"""