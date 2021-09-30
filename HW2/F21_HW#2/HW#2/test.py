import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('./HW2_LR.csv')
print(df.head(5))
new_df = df[['b', 'y']]



def calcBVec(data, w):
    x = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    b_vec = []
    b0 = []
    for xn in x:
        x0 = x[xn]
        xwt_sum = 0
        ywt_sum = 0
        w_sum = w.sum()
        for i in range(len(y)):
            xwt_sum += w[i] * x0[i]
            ywt_sum += w[i] * y[i]

        xw_bar = xwt_sum / w_sum
        yw_bar = ywt_sum / w_sum
        x_bar = x0.sum() / len(x0)
        y_bar = y.sum() / len(y)
        t_sum = 0
        b_sum = 0
        for i in range(len(y)):
            xi = x0[i]
            yi = y[i]
            t_sum += w[i]*(xi - xw_bar)*(yi - yw_bar)
            b_sum += w[i]*((xi - xw_bar)**2)
        b_vec.append(t_sum/b_sum)
        b0.append(yw_bar - b_vec[0]*xw_bar)
        
        v =5

    LR = LinearRegression()
    LR.fit(x, y)
    b0_sum = (b0[0] + b0[1])/2
    return b_vec

calcBVec(df[['b', 'c', 'y']], np.array([1]*10000))
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
