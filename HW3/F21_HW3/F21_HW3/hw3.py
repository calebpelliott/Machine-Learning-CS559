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
df = pd.read_csv('./HW3/F21_HW3/F21_HW3/F21_CS559_HW3_data.csv')
print(df.head(5))






from random import seed
from random import random
from random import randrange









def DecisionTree(df):
    X = df.drop('class', 1)
    y = df['class']
    for x in df.columns[0:-1]:
        print(x)
        #Cut by mean
        mean = df[x].mean()
        min = df[x].min()
        max = df[x].max()
        mid = (max + min) /2

        df[x] = pd.cut(df[x], bins=[min, mid, max], labels=[0,1])
    print(df.head(5))
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

    feat_order = []
    pred = {}

    node = getSortedFeatureScores(df)
    feat_order.append(node[0][0])
    l_df = df.loc[df[node[0][0]] == 0]
    r_df = df.loc[df[node[0][0]] == 1]
    #       c
    #     0   1
    #   a       a
    #  0 1     0 1
    # b   b    b  b
    #0 1 0 1  0 1 0 1
    #2 2 2 2  1 1 1 2
    #process left
    n_df = l_df.drop(node[0][0],1)
    node2 = getSortedFeatureScores(n_df)
    ll_df = n_df.loc[n_df[node2[0][0]] == 0]
    ll_df = ll_df.drop(node2[0][0],1)
    rr_df = n_df.loc[n_df[node2[0][0]] == 1]
    rr_df = rr_df.drop(node2[0][0],1)
    # percentage of b1 -> class 1 vs class 2
    total = len(ll_df)
    b0 = ll_df[ll_df[node2[1][0]] ==0]
    print(b0.head(5))
    pc1 = len(b0[b0['class'] == 1]) / len(b0)
    pc2 = len(b0[b0['class'] == 2]) / len(b0)
    if pc1 > pc2:
        pred['000'] = 1
    else:
        pred['000'] = 2

    b1 = ll_df[ll_df[node2[1][0]] ==1]
    print(b1.head(5))
    pc1 = len(b1[b1['class'] == 1]) / len(b1)
    pc2 = len(b1[b1['class'] == 2]) / len(b1)
    if pc1 > pc2:
        pred['001'] = 1
    else:
        pred['001'] = 2

    b0 = rr_df[rr_df[node2[1][0]] ==0]
    print(b0.head(5))
    pc1 = len(b0[b0['class'] == 1]) / len(b0)
    pc2 = len(b0[b0['class'] == 2]) / len(b0)
    if pc1 > pc2:
        pred['010'] = 1
    else:
        pred['010'] = 2

    b1 = rr_df[rr_df[node2[1][0]] ==1]
    print(b1.head(5))
    pc1 = len(b1[b1['class'] == 1]) / len(b1)
    pc2 = len(b1[b1['class'] == 2]) / len(b1)
    if pc1 > pc2:
        pred['011'] = 1
    else:
        pred['011'] = 2


    #process right
    n_df = r_df.drop(node[0][0],1)
    node2 = getSortedFeatureScores(n_df)
    ll_df = n_df.loc[n_df[node2[0][0]] == 0]
    ll_df = ll_df.drop(node2[0][0],1)
    rr_df = n_df.loc[n_df[node2[0][0]] == 1]
    rr_df = rr_df.drop(node2[0][0],1)
    # percentage of b1 -> class 1 vs class 2
    total = len(ll_df)
    b0 = ll_df[ll_df[node2[1][0]] ==0]
    print(b0.head(5))
    pc1 = len(b0[b0['class'] == 1]) / len(b0)
    pc2 = len(b0[b0['class'] == 2]) / len(b0)
    if pc1 > pc2:
        pred['100'] = 1
    else:
        pred['100'] = 2

    b1 = ll_df[ll_df[node2[1][0]] ==1]
    print(b1.head(5))
    pc1 = len(b1[b1['class'] == 1]) / len(b1)
    pc2 = len(b1[b1['class'] == 2]) / len(b1)
    if pc1 > pc2:
        pred['101'] = 1
    else:
        pred['101'] = 2

    b0 = rr_df[rr_df[node2[1][0]] ==0]
    print(b0.head(5))
    pc1 = len(b0[b0['class'] == 1]) / len(b0)
    pc2 = len(b0[b0['class'] == 2]) / len(b0)
    if pc1 > pc2:
        pred['110'] = 1
    else:
        pred['110'] = 2

    b1 = rr_df[rr_df[node2[1][0]] ==1]
    print(b1.head(5))
    pc1 = len(b1[b1['class'] == 1]) / len(b1)
    pc2 = len(b1[b1['class'] == 2]) / len(b1)
    if pc1 > pc2:
        pred['111'] = 1
    else:
        pred['111'] = 2

    pred_test = []
    #do predictions
    for index, row in df.iterrows():
        key = str(row['c']) + str(row['a']) + str(row['b'])
        try:
            pred_test.append(pred[key])
        except:
            pred_test.append(1)
    score = accuracy_score(pred_test, y)
    return score
selfscore = DecisionTree(df)




# Create a random subsample from the dataset with replacement
def buildRandomForest(df):
    ratio = .5
    rand_i = []
    needed= round(len(df) * ratio)

    for i in range(needed):
        rand_i.append(randrange(len(df)))
    new_df = df.iloc[rand_i,:]
    return new_df

def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample
df = pd.read_csv('./HW3/F21_HW3/F21_HW3/F21_CS559_HW3_data.csv')
l = len(df)
for _ in range(3):
    rforest = buildRandomForest(df)
    r_score = DecisionTree(rforest)
    print(r_score)
rand_df = buildRandomForest(df) 
x=3