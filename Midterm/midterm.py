import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_error






### code starts here
df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/Midterm/CS559_F21_Midterm/LR_SVM.csv")
X = DataFrame()
X['x1'] = df['x1'] 
X['x2'] = df['x2'] 
print(X)
X= X.to_numpy()
w=np.array([[1.199],[0.822]])
b= -5.653
pred = (X.dot(w) + b)
pred = [-1 if x <= .5 else 1 for x in pred]








df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/Midterm/CS559_F21_Midterm/Probability_Classification.csv")

df_0 = df[df['class'] == 0]
df_1 = df[df['class'] == 1]
print(df_0.var()['x'])
print(df_1.var()['x'])

E_x_0 = df_0.mean()['x']
E_x_1 = df_1.mean()['x']
df.var()




def my_gradient(X, y):
    n = .001
    e = 1
    num_samps = len(X)
    m = np.random.random(1)[0]
    b = np.random.random(1)[0]
    w = None
    while(e > .05):
        temp = m*X + b
        diff = y-temp
        
        m_deriv = X.dot(diff)
        m_deriv = m_deriv.sum() / num_samps
        m = m - (n * (-2 * m_deriv))
        
        b_deriv = diff.sum() / num_samps
        b = b - (n *(-2 * b_deriv))
        updated = m*X + b
        mse = mean_squared_error(y, updated)#((y - updated)**2).mean(axis=0)
        e = mse
        if mse < .05:
            w = [b, m]
    return w

df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/Midterm/CS559_F21_Midterm/gradient_question.csv")
w= my_gradient(df['x'], df['y'])