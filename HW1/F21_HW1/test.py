import numpy as np
from numpy.lib.function_base import delete
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA


np.random.seed(2342)
mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 30).T
assert class1_sample.shape == (3,30), "The matrix has not the dimensions 3x30"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 30).T
assert class2_sample.shape == (3,30), "The matrix has not the dimensions 3x30"

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

#plt.show()

#1
allx = np.append(class1_sample[0,:],  class2_sample[0,:], 0)
ally = np.append(class1_sample[1,:],  class2_sample[1,:], 0)
allz = np.append(class1_sample[2,:],  class2_sample[2,:], 0)
t_all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
all_samples = np.column_stack((allx, ally, allz))

t_mean_x = np.mean(t_all_samples[0,:])
t_mean_y = np.mean(t_all_samples[1,:])
t_mean_z = np.mean(t_all_samples[2,:])

#2
mean_x = np.mean(all_samples[:,0])
mean_y = np.mean(all_samples[:,1])
mean_z = np.mean(all_samples[:,2])
mean_vector = [[mean_x], [mean_y], [mean_z]]
mean_vector = np.array(mean_vector)
t_mean_vector = np.array([[mean_x], [mean_y], [mean_z]])


#3
t_sm = np.zeros((3,3))
for i in range(t_all_samples.shape[1]):
    t_sm += (t_all_samples[:,i].reshape(3,1) - t_mean_vector).dot((t_all_samples[:,i].reshape(3,1) - t_mean_vector).T)
    s1 = t_all_samples[:,i].reshape(3,1)
    s1 = s1 - t_mean_vector
    s2 = s1.dot((s1).T)
    sf = (t_all_samples[:,i].reshape(3,1) - t_mean_vector).dot((t_all_samples[:,i].reshape(3,1) - t_mean_vector).T)
coun = 0
sm = np.zeros((3,3))
for i in all_samples:
    intmd = np.array([[i[0]], [i[1]], [i[2]]])
    intmd = intmd - mean_vector
    intmd = intmd.dot((intmd).T)
    sm = sm + intmd

#4
t_eig_val_sc, t_eig_vec_sc = np.linalg.eig(t_sm)

#Need eig_vec_sc for plot below
eigen_val, eig_vec_sc = np.linalg.eig(sm)

#5
# Make a list of (eigenvalue, eigenvector) tuples
t_eig_pairs = [(np.abs(t_eig_val_sc[i]), t_eig_vec_sc[:,i]) for i in range(len(t_eig_val_sc))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
t_eig_pairs.sort(key=lambda x: x[0], reverse=True)

eigen_list = []
for i in range(len(eigen_val)):
    eigen_list.append((np.abs(eigen_val[i]), eig_vec_sc[:,i]))
eigen_list.sort(key = lambda x: x[0])
eigen_list.reverse()

#7
t_matrix_w = np.hstack((t_eig_pairs[0][1].reshape(3,1), t_eig_pairs[1][1].reshape(3,1)))
a1 = np.array([np.array([eigen_list[0][1][0], eigen_list[1][1][0]])])
a2 = np.array([np.array([eigen_list[0][1][1], eigen_list[1][1][1]])])
a3 = np.array([np.array([eigen_list[0][1][2], eigen_list[1][1][2]])])
w = np.array([a1, a2, a3])




np.random.seed(2342)
mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 30).T
assert class1_sample.shape == (3,30), "The matrix has not the dimensions 3x30"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 30).T
assert class2_sample.shape == (3,30), "The matrix has not the dimensions 3x30"

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

#plt.show()

allx = np.append(class1_sample[0,:],  class2_sample[0,:], 0)
ally = np.append(class1_sample[1,:],  class2_sample[1,:], 0)
allz = np.append(class1_sample[2,:],  class2_sample[2,:], 0)

xyz = np.column_stack((allx, ally, allz))
df = pd.DataFrame(xyz)
std = df.loc[:].values
std = StandardScaler().fit_transform(std)
print(df.head(5))
# Standardizing the features
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(std)
principalDf = pd.DataFrame(data=principalComponents, columns=['p1', 'p2'])

final = ['class1']*class1_sample.shape[1] + ['class2']*class2_sample.shape[1]
principalDf['target'] = final
print(principalDf.head(5))
print(principalDf.tail(5))

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['class1', 'class2']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['target'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'p1']
               , principalDf.loc[indicesToKeep, 'p2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()










def calculateCentroids(map):
    for item in map:
        avgx = sum(map[item]['dp'][:,0]) / len(map[item]['dp'])
        avgy = sum(map[item]['dp'][:,1]) / len(map[item]['dp'])
        map[item]['centroid'] = np.array([avgx, avgy])
    return map

def regroupPoints(map):
    newMap = {}
    for item in map:
        newMap[item] = {'dp' :None, 'centroid':map[item]['centroid']}
    changeOccured = False
    for item in map:
        for dp in map[item]['dp']:
            shortest = None
            newGroup = None
            for cent in map:
                dist = np.linalg.norm(dp - map[cent]['centroid'])
                if shortest is None:
                    shortest = dist
                    newGroup = cent
                elif dist < shortest:
                    shortest = dist
                    newGroup = cent

            if item != newGroup:
                changeOccured = True
            if newMap[newGroup]['dp'] is None:
                newMap[newGroup]['dp'] = [dp]
            else:    
                newMap[newGroup]['dp'] = np.append(newMap[newGroup]['dp'], [dp],0)
    return changeOccured, newMap

def MY_KMeans(n, data):
    groupMap = {}

    #Iterate through data and assign random data points to groups
    count = 0
    for i in range(len(data)):
        idx = np.random.randint(0, len(data))
        dp = data[idx]
        data = np.delete(data,idx,0)
        if not count in groupMap.keys():
            groupMap[count] = {'dp' :[dp], 'centroid':None}
        else:
            points = np.append(groupMap[count]['dp'], [dp], axis=0)
            groupMap[count]['dp'] = points
        count = (count + 1) % n

    converged = False
    while not converged:
        groupMap = calculateCentroids(groupMap)
        converged, groupMap = regroupPoints(groupMap)

    return groupMap

data = []
e = np.array([[.5,.5]])
e = np.append(e, [[.1,.1]], axis=0)
data = np.random.randint(-10, 10, size=(100, 2))

MY_KMeans(3, data)

Data = pd.read_csv('HW1_Q1_0.csv',encoding='ISO-8859-1')
Data.columns = ['X', 'Y']

kmeans = KMeans()
kmeans.set_params(n_clusters=5)
kmeans.fit(Data)

def flip_coin(prob, times, numCoins, numOccur):
    num = 0
    for trial in range(times):
        numHeads = 0
        for coin in range(numCoins):
            if prob < np.random.rand():
                numHeads += 1
        if numHeads == numOccur:
            num += 1
    return num

gdp = pd.read_csv('GDP.csv', skiprows=range(5), header= None, usecols=[0,1,3,4], nrows=190,encoding='ISO-8859-1')
gdp.columns = ['CountryCode', 'Rank', 'CountryName', 'GDP']
Country = pd.read_csv('Country.csv',encoding='ISO-8859-1')

dc = pd.DataFrame({'A' : [1, 2, 3, 4],'B' : [4, 3, 2, 1],'C' : [3, 4, 2, 2]})
plt.plot(dc)
plt.show


g = list(range(50, 10050, 50))
e = len(g)
q = list(range(1, 200))
pi = .5
n = 20
k = 10
w = np.random.binomial(9, 0.1, 20000)# == 0
s = []
for x in g:
    #result = flip_coin(pi, x, n, k)
    tests = np.random.binomial(n,pi,x)
    succ = tests == k
    succ_tot = sum(succ)
    succ_prob = succ_tot/x
    p = sum(np.random.binomial(n,pi,x) == k)
    s.append(p/x)
ProbTable=pd.DataFrame()
ProbTable.t = g 
ProbTable.s = s 
z = sum(s) / len(s)
plt.plot(ProbTable.t, ProbTable.s, 'g-')
plt.axhline(y=z, color='r', linestyle='--')
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Probability vs. ')
plt.show()




s = pd.merge(gdp, Country, on="CountryCode")
B = s
B.GDP = B.GDP.apply(lambda x : x.replace(',',''))
print(B)
B = B.astype({"GDP":np.int64})
num = 0
print(B["GDP"])
for x in B["GDP"]:
    num += x
    print(x)
num = num / 189
AVG = B["GDP"].mean()
print(B)
REG = B.groupby(["Region"]).agg({"GDP" : {'mean', np.std}})
print(REG)
print(REG.agg({'mean'}))
for i in s["GDP"]:
    print(i)
J = s.groupby('Region').mean()
print(J.columns)
#print(s.columns)
#print(len(gdp.CountryCode))
#print(len(Country.CountryCode))
#print(len(s.CountryCode))
x = s["Region"].value_counts()
#s['Region'].value_counts().plot(kind='bar')
x = s["GDP"]

x = s.groupby('Region')
print(s)
vv = s["Region"].unique().tolist()

cc = [38.0, 2.0, 18.0, 22.0, 21, 0,0]*7
df = pd.DataFrame([[38.0, 2.0, 18.0, 22.0, 21, 0,0],[38.0, 2.0, 18.0, 22.0, 21, 0,0],[38.0, 2.0, 18.0, 22.0, 21, 0,0],[38.0, 2.0, 18.0, 22.0, 21, 0,0],[38.0, 2.0, 18.0, 22.0, 21, 0,0],[38.0, 2.0, 18.0, 22.0, 21, 0,0],[38.0, 2.0, 18.0, 22.0, 21, 0,0]],
                  index=pd.Index( vv, name='Regions'),
                  columns=pd.Index(vv))
print(df)


for i in s["Region"].unique():
    print(i)
y = x.get_group("Europe & Central Asia")
d = y["GDP"]
base = []
for i in s["Region"].unique():
    new = []
    for j in s["Region"].unique():
        r1 = x.get_group(i)
        r2 = x.get_group(j)
        test= r1["GDP"]
        test2= r2["GDP"]
        p = ks_2samp(r1["GDP"],r2["GDP"])
        sting = '(%0.5f' % p[0]
        sting += ', %0.5f)' % p[1]
        new.append(sting)
    base.append(new)

df = pd.DataFrame(base,
                  index=pd.Index( vv, name='Regions (stat, p)'),
                  columns=pd.Index(vv))
print(df)


x = ks_2samp(d,d)
print(y)
for i in y:
    print(i)
x = ks_2samp(s["Region"],s["GDP"])
print(x)
o=0
for i in x:
    print(i)
    o+=i
x=3