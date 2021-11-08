import pandas as pd
import numpy as np
import math
import  matplotlib.pyplot as plt

x = np.linspace(start=.001, stop=.999, num=100)  # math.log won't accept 0 or 1
y1 = list(map(lambda i: 2 * i * (1 - i), x))
y2 = list(map(lambda t: -t * math.log(t, 2) - (1 - t) * math.log((1 - t), 2), x))

plt.plot(x, y1, label='gini')
plt.plot(x, y2, label='entropy')
plt.legend(loc=1)
#plt.show()

filename = "auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
df = pd.read_csv(filename, delim_whitespace=True, names=column_names)

df.isnull().sum() / df.shape[0] * 100.00

df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')

mpg_cut=(df.mpg.max()+df.mpg.min())/2

df['year']=df['year'].astype(float)
df['mpg'] = pd.cut(df.mpg, bins=[df.mpg.min()-1,mpg_cut,df.mpg.max()], labels=[0,1])

def value_cut(data,feature):
    for feat in feature:
        feat_max=data[feat].max()
        feat_min=data[feat].min()
        feat_cut1=(feat_max+feat_min)/3
        data[feat]=pd.cut(df[feat],bins=[feat_min,feat_cut1,feat_cut1*2,feat_max],labels=[0,1,2])
        print(feat,":",feat_min,feat_cut1,2*feat_cut1,feat_max)
        
    return data

value_cut(df,['displacement','acceleration','weight','horsepower'])

yearmin=df.year.min()
yearmax=df.year.max()
x=(yearmax-yearmin)/4
print(yearmin,yearmax,x)
yearlist=[yearmin-1,yearmin+4,yearmin+8,yearmax]
print(yearlist)

df['year'] = pd.cut(df.year, bins=yearlist, labels=[0,1,2])

from collections import Counter
import math

def purity(L, metric='gini'):
    total = len(L)
    freq = map(lambda x: float(x) / total, list(Counter(L).values()))
    a = list(freq)
    a = [a[0],a[1]]
    if metric == 'gini':
        scores = map(lambda x: x * (1 - x), freq)
        aa = [x*(1-x) for x in a]
    elif metric == 'entropy':
        scores = map(lambda x: -x * math.log(x, 2), freq)
    asum = sum(aa)
    return sum(scores)

df = pd.read_csv('./F21_CS559_HW3_data.csv')
for x in df:
    print(x)
    #Cut by mean
    mean = df[x].mean()
    min = df[x].min()
    max = df[x].max()
    mid = (max + min) /2

    df[x] = pd.cut(df[x], bins=[min, mid, max], labels=[0,1])

for feat in ['a','b','c']:
    L=list(df[feat])
    print(feat,purity(L))