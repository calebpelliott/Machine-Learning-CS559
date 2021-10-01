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

kmeans = KMeans()
kmeans.set_params(n_clusters=6)
kmeans.fit(new_df)

guess = []
for i in range(len(new_df)):
    dp = new_df.loc[[i]]#.values
    c = kmeans.predict(dp)
    guess.append(c[0])
    x = 3
class_guesses = []
count = 0
for i in range(6):
    class_guesses.append(guess[count : count + 1250])
    count += 1250

for guess in class_guesses:
    mode = max(set(guess), key=guess.count)
    right = guess.count(mode)
    wrong = 1250 - right
    wg = guess[-1]
cl = guess[0:1250]
c2 = guess[1250:2500]
c3 = guess[1250:2500]
c4 = guess[1250:2500] 
c5 = guess[1250:2500]
c6 = guess[1250:2500]
guess = np.array(guess).reshape(-1,1)
fin = df[['Class']]
fin['pred'] = guess

fin.to_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/PRJ1/F21/predict.csv")
