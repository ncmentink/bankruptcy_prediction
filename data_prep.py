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


#Data transformation
# TO DO:
# VARIABLE NAMES AANPASSEN
# ZOEKEN NAAR TE HOGE CORRELATIES TUSSEN VARIABELEN: MULTICOLLINEARTIY
# TESTEN VOOR CORRELATIE MET DE Y



#Nog aanpassen
"""
# One-hot encoding for categorical variables

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


# Create correlation plot
data_corr = data.drop(columns=['term', "term_36", "term_60", 'sub_grade_A', 'sub_grade_B', 'sub_grade_C', 'sub_grade_D', 'sub_grade_E',
                               'sub_grade_F', 'sub_grade_G'
                               ])
print(data_corr.corr())

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
plot_confusion_matrix(data.corr())


""""
# Write data to csv after transformations
data_final.to_csv('data_final.csv', index=False)
"""