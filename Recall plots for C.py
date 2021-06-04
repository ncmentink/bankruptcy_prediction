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

X_plot = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3, 4, 5, 6, 7, 8, 9, 10]
y_plot_L1 = [0.0, 0.0, 0.1318181818181818, 0.37272727272727274, 0.4772727272727273, 0.5227272727272727, 0.5590909090909091, 0.5909090909090909, 0.6045454545454545, 0.6136363636363636, 0.7090909090909091, 0.7181818181818181, 0.7227272727272728, 0.7272727272727273, 0.7363636363636363, 0.759090909090909, 0.75, 0.75, 0.7454545454545455, 0.740909090909091, 0.7136363636363636, 0.7545454545454545, 0.7318181818181818, 0.7318181818181818, 0.7272727272727273, 0.7136363636363636, 0.7181818181818181, 0.7, 0.7136363636363636, 0.7318181818181818, 0.6727272727272727, 0.6954545454545454, 0.6954545454545454, 0.7181818181818181, 0.7, 0.7136363636363636, 0.7045454545454546, 0.6818181818181818, 0.6909090909090909, 0.7045454545454546, 0.7, 0.7136363636363636, 0.7136363636363636, 0.7, 0.7136363636363636, 0.7]
y_plot_L2 = [0.6090909090909091, 0.6636363636363637, 0.6909090909090909, 0.6954545454545454, 0.6863636363636364, 0.7045454545454546, 0.6909090909090909, 0.7136363636363636, 0.6636363636363637, 0.6954545454545454, 0.7045454545454546, 0.7272727272727273, 0.6909090909090909, 0.7363636363636363, 0.7318181818181818, 0.7454545454545455, 0.7227272727272728, 0.7, 0.7318181818181818, 0.7136363636363636, 0.7363636363636363, 0.7454545454545455, 0.7181818181818181, 0.7181818181818181, 0.6772727272727272, 0.7090909090909091, 0.6954545454545454, 0.7272727272727273, 0.7090909090909091, 0.6909090909090909, 0.7090909090909091, 0.6954545454545454, 0.7136363636363636, 0.6954545454545454, 0.7136363636363636, 0.6909090909090909, 0.6954545454545454, 0.7045454545454546, 0.7, 0.6909090909090909, 0.6818181818181818, 0.6818181818181818, 0.7045454545454546, 0.7, 0.7227272727272728, 0.7]



fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot()
plt.xlabel('Different values of C', fontsize=18)
plt.ylabel('Recall', fontsize=16)
ax1.plot(X_plot, y_plot_L1, marker='o',label='Lasso')
ax1.plot(X_plot, y_plot_L2, color='olive', marker='o',label='Ridge')
plt.legend(loc="lower right", prop={'size': 20})
ax2 = fig.add_subplot(324)
ax2.plot()
plt.title("Zoomed in for C = [0.01, 0.7]")
ax2.plot(X_plot[10:25], y_plot_L1[10:25], marker='o',label='Lasso')
ax2.plot(X_plot[10:25], y_plot_L2[10:25], color='olive', marker='o',label='Ridge')

plt.show()
