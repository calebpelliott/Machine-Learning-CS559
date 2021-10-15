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
    return k

def plot_3D(df, comp1,comp2,comp3,target):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[comp1],df[comp2],df[comp3],c=df[target])
    ax.set_xlabel(comp1)
    ax.set_ylabel(comp2)
    ax.set_zlabel(comp3)
    plt.show()



df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/PRJ1/F21/H.csv")






new_df = df[['var1', 'var2', 'var3', 'var4','var5', 'var6', 'var7', 'var8', 'var9']]
#new_df = df[['var1', 'var2', 'var3']]
std = new_df.loc[:].values
std = StandardScaler().fit_transform(std)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(std)
#principalDf = pd.DataFrame(data=principalComponents, columns=['p1', 'p2'])
principalDf = pd.DataFrame(data=principalComponents, columns=['p1', 'p2', 'p3'])

principalDf['target'] = df['Class']
plot_3D(principalDf,'p1','p2','p3','target')
print(principalDf.head(5))


# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component sklearn PCA', fontsize = 20)
# targets = [1, 2, 3, 4, 5, 6]
# colors = ['r', 'g', 'b', 'y', 'm', 'c']
# for target, color in zip(targets,colors):
#     indicesToKeep = principalDf['target'] == target
#     ax.scatter(principalDf.loc[indicesToKeep, 'p1']
#                , principalDf.loc[indicesToKeep, 'p2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()

kmeans = KMeans()
def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(n_cluster_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    tabular_result = pd.DataFrame(inertias, 
                                  index = ["n = {}".format(i) for i in range(1, len(inertias)+1)], 
                                  columns=['Inertia'])
    
    return tabular_result

#new_df = principalDf[['p1', 'p2']]
new_df = principalDf[['p1', 'p2', 'p3']]
plot_inertia(kmeans, new_df, range(1, 10))

kmeans = KMeans()
kmeans.set_params(n_clusters=6)
kmeans.fit(new_df)

km_guess = kmeans.predict(new_df)

p = calcK(km_guess)

nn = pd.get_dummies(df.Class, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
#nn = nn.rename(columns={'c1', 'c2', 'c3', 'c4', 'c5', 'c6'})
newdf = df.join(nn)

newdf = newdf[['var4','var5','var6','var7','var8','var9',1,2,3,4,5,6]]
corr = newdf.corr()
corr.style.background_gradient(cmap='coolwarm')

c1 = df[df['Class'] == 1]#np.array([1]*1249 + [0]*(7500-1249)).reshape(-1,1)
counts = df["Class"].value_counts()
corr = df.corr()
#print(corr['c1'].sort_values(ascending=False))







new_df = df[['var1', 'var2']]
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







accuracy = []
below95 = False
k = 1
while not below95:
    knn = KNeighborsClassifier(k)
    knn.fit(new_df, df.iloc[:,-1])
    knn_guess = knn.predict(new_df)
    accuracy.append(calcK(knn_guess))
    knn_guess = calcK(knn_guess)
    if knn_guess < 95:
        below95 = True
    k += 10
    print(k)
calcK(km_guess)
calcK(knn_guess)
calcK(lr_guess)
