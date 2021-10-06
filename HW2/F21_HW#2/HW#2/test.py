import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/HW2/F21_HW#2/HW#2/HW2_LR.csv")
print(df.head(5))
new_df = df[['b', 'y']]

peas = np.array([[0.21, 0.1726], 
                [0.2, 0.1707], 
                [0.19, 0.1637],
                [0.18, 0.164],
                [0.17, 0.1613],
                [0.16, 0.1617],
                [0.15, 0.1598]])

variance = np.var(peas, axis=1)
std = np.std(peas, axis=1)

np.random.seed(1)
N = 30
## generate a random array with 2 columns
beta = np.array([2, -7, 5])
x_m = np.random.randn(N, 2)
y_m = np.dot(np.append(np.ones((N,1)), x_m, axis=1), beta) + np.random.randn(N)*4

## transform x to a matrix
x_m = np.mat(x_m)
## y transpose converts y into a column vector
y_m = np.mat(y_m).T
## add a column of 1’s to x
x_m = np.column_stack((np.ones([x_m.shape[0], 1]), x_m))
## matrix.I returns the inverse of that matrix
B = (x_m.T*x_m).I * x_m.T * y_m


def calcWvec(data, r_n):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:,-1]
    size = y.size
    ones = np.array([1]*size).reshape(-1,1)
    x_m = np.mat(np.append(ones, x.to_numpy(),1))
    y_m = np.mat(y.to_numpy().reshape(-1,1))
    #make diaganol of r
    w = np.mat(np.diagflat(r_n))

    B = (x_m.T*w*x_m).I * x_m.T  * w * y_m
    return B.T

def my_error(data, r_n):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:,-1]
    size = y.size
    ones = np.array([1]*size).reshape(-1,1)
    x_m = np.mat(np.append(ones, x.to_numpy(),1))
    y_m = np.mat(y.to_numpy().reshape(-1,1))
    w = calcWvec(data, r_n)
    
    e = 0
    for i in range(len(y_m)):
        t_n = y_m[i]
        x_n = x_m[i]
        est = w * x_n.T
        e_n = r_n[i] * (t_n - est)**2
        e += e_n
    return e/2
faer = 44

df.hist()
#plt.show()
uniform_df=df[['a', 'b', 'c']]
scaler = QuantileTransformer(output_distribution='normal')
df_scale= scaler.fit_transform(uniform_df)
df['a'] = df_scale[:,0].reshape(-1,1)
df['b'] = df_scale[:,1].reshape(-1,1)
df['c'] = df_scale[:,2].reshape(-1,1)

rskew_df=df[['k']]
u = rskew_df.to_numpy().min()
a = rskew_df.to_numpy()
rskew_df += np.abs(rskew_df.to_numpy().min())
a = rskew_df.to_numpy()
rskew_df= np.power((rskew_df),1/3)
df['k'] = rskew_df
rskew_df.hist()
#plt.show()

lskew_df=df[['d']]
u = lskew_df.to_numpy().min()
lskew_df += np.abs(lskew_df.to_numpy().min())
lskew_df.hist()
#plt.show()
lskew_df= np.power((lskew_df),2)
lskew_df.hist()
#plt.show()
df['d'] = lskew_df


scaler = StandardScaler()
scale = scaler.fit_transform(df)
f_df = DataFrame(scale, columns=['a','b','c','d','k','y'])
print(f_df.head(5))

X = f_df[['b']]
y = f_df[['y']]
### Linear Regression Modeling starts here
LR = LinearRegression()
LR.fit(X, y)

y_train_pred = LR.predict(X)
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f' %(mean_squared_error(y,y_train_pred)))
#                                     mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train: %.3f' %(r2_score(y,y_train_pred)))
#                                     r2_score(y_test,y_test_pred)))


f_df.hist()
plt.show()

df.hist()
plt.show()





data = df[['b', 'y']]
data.hist()
plt.show()
print(data.head(5))
db = data[['b']]
dy = data[['y']]

plt.hist(db, color='green', bins=100)
plt.show()

scaler = QuantileTransformer(output_distribution='normal')
db_scale = scaler.fit_transform(db)

plt.hist(db_scale, color='green', bins=100)
plt.show()

data['b'] = db_scale
data.hist()
plt.show()

scaler = StandardScaler()
scale = scaler.fit_transform(data)
data['b'] = scale[:,0]
data['y'] = scale[:,1]
data.hist()
plt.show()

y = scale[:,1].reshape(-1,1)
X = scale[:,0].reshape(-1,1)

LR = LinearRegression()
LR.fit(X, y)

y_train_pred = LR.predict(X)
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f' %(mean_squared_error(y,y_train_pred)))
#                                     mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train: %.3f' %(r2_score(y,y_train_pred)))
#                                     r2_score(y_test,y_test_pred)))
data = data.sort_values('y')
print(data.head(5))
error = my_error(df[['b', 'y']], np.array([1]*10000).reshape(-1,1))
variance = np.var(x_m, axis=1)
a = np.reciprocal(variance)
aa = np.diagflat(a)

## weighted
w = np.diag(r_n)
w = np.diagflat(r_n)#np.mat(w)
x_m = np.mat(x_m)

## y transpose converts y into a column vector
y_m = np.mat(y_m).T
## add a column of 1’s to x
x_m = np.column_stack((np.ones([x_m.shape[0], 1]), x_m))
## matrix.I returns the inverse of that matrix
B = (x_m.T*w*x_m).I * x_m.T  * w * y_m
B = B.T
e = 0
for i in range(len(y_m)):
    t_n = y_m[i]
    x_n = x_m[i]
    est = B * x_n.T
    e_n = r_n[i] * (t_n - est)**2
    e += e_n
faer = 44
def calcWvec(data, r_n):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:,-1]
    
    xnp = x.to_numpy()
    ynp = y.to_numpy().reshape(-1,1)

    #r_n needs to be made into diagonal
    r_diag = np.diag(r_n)
    if len(xnp.shape) == 1:
        xnp = xnp.reshape(-1,1)

    xx = xnp.T
    t = xx*r_diag
    ww = xnp.dot(r_diag)
    W = np.linalg.inv(xnp.T.dot(r_diag).dot(xnp)).dot(xnp.T).dot(r_diag).dot(ynp)
    W = np.linalg.inv(xnp.T.dot(r_n*xnp)).dot(xnp.T).dot(r_n*ynp)
    return W

def my_error(data, r_n):
    y = data.iloc[:,-1]
    x = data.iloc[:,0]

    xnp = x.to_numpy().reshape(-1,1)
    ynp = y.to_numpy().reshape(-1,1)

    w = calcWvec(data, r_n)
    A = (y - w.T.dot(x))
    error = .5 * r_n

def calcBVec(data, w):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:,-1]
    bb = np.array([1] * 5000 + [2] * 5000)
    LR = LinearRegression()
    w = bb.reshape(-1, 1)
    LR.fit(x,y, bb)
    xnp = x.to_numpy()#.reshape(-1,1)
    ynp = y.to_numpy().reshape(-1,1)
    if len(xnp.shape) == 1:
        xnp = xnp.reshape(-1,1)

    A = np.linalg.inv(xnp.T.dot(xnp)).dot(xnp.T).dot(ynp)
    BB = w*xnp
    B1 = np.linalg.inv(xnp.T.dot(w*xnp)).dot(xnp.T).dot(w*ynp)
    #B = np.linalg.inv(x.T.dot(w).dot(x)).dot(x.T).dot(w).dot(ynp)

    b_vec = []
    b0 = []
    for xn in x:
        x0 = x[xn]
        w_sum = w.sum()
        xw_bar = w.T.dot(x0) / w_sum
        yw_bar = w.T.dot(y) / w_sum
        x_bar = x0.sum() / len(x0)
        y_bar = y.sum() / len(y)
        t_sum = 0
        b_sum = 0
        for i in range(len(y)):
            xi = x0[i]
            yi = y[i]
            t_sum += w[i]*(xi - xw_bar)*(yi - yw_bar)
            b_sum += w[i]*((xi - xw_bar)**2)
        #tt_sum = (w * (x0 - xw_bar)).T.dot((y - yw_bar))
        #bb_sum = w.T.dot(w * (x0 - xw_bar)**2)
        b_vec.append(t_sum/b_sum)
        b0.append(yw_bar - b_vec[0]*xw_bar)
        
        v =5

    LR = LinearRegression()
    LR.fit(x, y)
    b0_sum = (b0[0] + b0[1])/2
    return b_vec

calcWvec(df[['b', 'c', 'y']], np.array([1,1]))
my_error(df[['b', 'y']], np.array([1]*10000).reshape(-1,1))
new_y = new_df['y'].values
new_df['y'] = new_y[new_y > -4000000]
scaler = QuantileTransformer()
f_df = scaler.fit_transform(new_df)
#print(f_df)
#plt.hist(f_df[:,1], color='green', bins=100)
#plt.show()
#X = f_df[:,0].reshape(-1, 1)
#y = f_df[:,1].reshape(-1, 1)
X = df[['b']]
y = df[['y']].values
qscale = QuantileTransformer()
X_scale = qscale.fit_transform(X)
scale = StandardScaler()
y_scale = scale.fit_transform(y)

#X = [[f_df[:,0].reshape(-1, 1)], ]#X_scale#
#y = f_df[:,1].reshape(-1, 1)#y_scale
df_y = df[['y']].values
df_y = df_y[df_y > -4000000]
plt.hist(df_y, color='green', bins=100)
plt.show()
y_map = np.sqrt(df[['y']])
plt.hist(y_map, color='green', bins=100)
plt.show()
X = f_df[:,0].reshape(-1, 1)
y = df_y
LR = LinearRegression()
LR.fit(X, y)

y_train_pred = LR.predict(X)
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f' %(mean_squared_error(y,y_train_pred)))
#                                     mean_squared_error(y_test,y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train: %.3f' %(r2_score(y,y_train_pred)))
#                                     r2_score(y_test,y_test_pred)))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]) 
np.random.seed(5)
x= np.array
print("ha")
x = np.array([[(3,2)], 
              [(4,5)]])
x = np.random.randint(1, 10, size=(5, 1))
beta = np.random.randint(1, 10, size=(5, 1))
xT = x.T
y = beta.dot(x.T)
print(y)

np.random.seed(18)
N = 10
X = np.random.rand(N)
X = np.array(sorted(X))

beta_0 = 1
beta_1 = 0.5
Y = beta_1*X + beta_0 
b=4
