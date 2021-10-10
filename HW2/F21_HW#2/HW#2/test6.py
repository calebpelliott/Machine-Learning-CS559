import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/HW2/F21_HW#2/HW#2/HW2_LR.csv")

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

### my_LR starts here
def my_LR(data,r_n):
    w = []
    error = []
    for r in r_n:
        w.append(calcWvec(data, r))
        error.append(my_error(data, r))
    
    return w, error, r_n

df = df[['b','y']]
df.sort_values(by=['b'], inplace=True)
print(df.head(5))
w,e,r = my_LR(df, [np.array([1]*10000).reshape(-1,1)])
st = e.values
w,e1,r = my_LR(df, [np.array([.1]*10000).reshape(-1,1)])
w,e2,r = my_LR(df, [np.array([.5]*10000).reshape(-1,1)])
w,e3,r = my_LR(df, [np.array([.1]*4000 + [.9]*2000 + [.1]*4000).reshape(-1,1)])
w,e4,r = my_LR(df, [np.array([.9]*4000 + [.1]*2000 + [.9]*4000).reshape(-1,1)])
x = 3