import numpy as np
from numpy.lib.function_base import delete
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans

a = np.array([[1,2,3],[4,5,6]]) 


a = np.append(a, [[7,8,9]],0) 



def MY_KMeans(n, data):
    groupMap = {}

    #Iterate through data and assign random data points to groups
    count = 0
    for i in range(len(data)):
        idx = 0#np.random.randint(0, len(data)
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
        calculateCentroids()
data = []
e = np.array([[.5,.5]])
e = np.append(e, [[.1,.1]], axis=0)
data = np.random.randint(-10, 10, size=(100, 2))
#for x in range(100):
#    data.append(np.random.random((2,2)))

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