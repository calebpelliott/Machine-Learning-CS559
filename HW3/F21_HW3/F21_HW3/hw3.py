from typing import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('./F21_CS559_HW3_data.csv')
print(df.head(5))


plt.hist(df['c'])
#plt.show()
print(df.dtypes)

aa = df.columns
for x in df.columns[0:-1]:
    print(x)
    #Cut by mean
    mean = df[x].mean()
    min = df[x].min()
    max = df[x].max()
    mid = (max + min) /2

    df[x] = pd.cut(df[x], bins=[min, mid, max], labels=[0,1])

def calcGini(item):
#Calculate gini
    count = Counter(item)
    total = len(item)
    count_list = [count[0], count[1]]
    freqs = [float(x)/total for x in count_list]
    gini_scores = [x*(1-x) for x in freqs]
    final_score = sum(gini_scores)
    return final_score

def getSortedFeatureScores(df):
    scores = {}
    for feat in df.columns[0:-1]:
        L=list(df[feat])
        scores[feat] = calcGini(L)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def checkEnd(df):
    #Check if we have reached an end
    length = len(list(new_df['class']))
    for x in new_df.columns[0:-1]:
        a = len(new_df[(new_df[x] == 0) & (new_df['class'] == 1)])
        b = len(new_df[(new_df[x] == 1) & (new_df['class'] == 2)])
        total = a+b
        if total == length:
            return True

    return False

def traverseTree(root, max_depth=3, current_depth=0):
    categories = [0,1]
    for cat in categories:
        new_df = root.loc[root[node[0][0]] == cat]
        new_df = new_df.drop(node[0][0], 1)


node = getSortedFeatureScores(df)
new_df = df.loc[df[node[0][0]] == 1]
new_df = new_df.drop(node[0][0], 1)
traverseTree(new_df)
new_df.to_csv('./c.csv')
x=3