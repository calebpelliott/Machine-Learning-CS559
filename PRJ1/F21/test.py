from numpy.lib.twodim_base import tri
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/PRJ1/F21/H.csv")

new_df = df[['var4', 'var5', 'var6', 'var7', 'var8', 'var9']]
result_df = df[['Class']]


kmeans = KMeans()
kmeans.set_params(n_clusters=6)
kmeans.fit(new_df)

knn = KNeighborsClassifier(15)
knn.fit(new_df, df.iloc[:,-1])

lr = LogisticRegression()
lr.fit(new_df, df.iloc[:,-1])
#knn.fit(new_df)
km_guess = kmeans.predict(new_df)
knn_guess = knn.predict(new_df)
lr_guess = lr.predict(new_df)
result_df['kmean'] = km_guess
result_df['knn'] = knn_guess

#result_df.to_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/PRJ1/F21/predict.csv")





def calcK(guess):
    count = 0
    class_guesses = []
    for _ in range(6):
        class_guesses.append(guess[count : count + 1250])
        count += 1250

    tright = 0
    for cg in class_guesses:
        cg = cg.tolist()
        mode = max(set(cg), key=cg.count)
        right = cg.count(mode)
        tright += right
        print(f"Group found: {mode}")
    k = tright / 7500 * 100
    print(f"Accuracy {k}%")

calcK(km_guess)
calcK(knn_guess)
calcK(lr_guess)
