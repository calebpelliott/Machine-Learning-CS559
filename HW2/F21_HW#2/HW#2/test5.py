import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('C:/Users/caleb/Documents/git/Machine-Learning-CS559/HW2/F21_HW#2/HW#2/HW2_LR.csv')
np.random.seed(8)
X = np.random.randn(1000,1)
y = 2*(X**3) + 10 + 4.6*np.random.randn(1000,1)

# Weight Matrix in code. It is a diagonal matrix.
def wm(point, X, tau): 
  # tau --> bandwidth
  # X --> Training data.
  # point --> the x where we want to make the prediction.
    
  # m is the No of training examples .
    m = X.shape[0] 
    
  # Initialising W as an identity matrix.
    w = np.mat(np.eye(m)) 
    
  # Calculating weights for all training examples [x(i)'s].
    for i in range(m): 
        xi = X[i] 
        d = (-2 * tau * tau) 
        w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d) 
        ww = w[i,i]
        if ww != 0:
            c=55
    return w
def error(w, X, y, r):
    total = 0
    for point in range(len(X)):
        # 1/2 * r(tn - W.T*xn)^2
        local_error = r * pow(y[point] - w.T*X,2)
        total += local_error
    return .5 * total

def predict(X, y, point, tau): 
    
   # m = number of training examples. 
    m = X.shape[0] 
    
   # Appending a cloumn of ones in X to add the bias term.
## # Just one parameter: theta, that's why adding a column of ones        #### to X and also adding a 1 for the point where we want to          #### predict. 
    X_ = np.append(X, np.ones(m).reshape(m,1), axis=1) 
    
   # point is the x where we want to make the prediction. 
    point_ = np.array([point, 1]) 
    
   # Calculating the weight matrix using the wm function we wrote      #  # earlier. 
    w = wm(point_, X_, tau)
    w_array =np.asarray(w)
    w = r_n
  # Calculating parameter theta using the formula.
    theta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y)) 
    
  # Calculating predictions.  
    pred = np.dot(point_, theta) 
    
    #for i in w:
        
    #calculate error
    t_error = 0
    for i in range(m):
        yi = y[i]
        xi = X_[i]
        e = theta.T.dot(xi)
        e = pow(yi - e,2)
        weight = w_array[i][i]
        err = weight * pow(yi-e,2)
        t_error += err
    t_error /= 2
   # Returning the theta and predictions 
    return theta, pred, t_error

def plot_predictions(X, y, tau, nval):   # X --> Training data. 
   # y --> Output sequence.
   # nval --> number of values/points for which we are going to
   # predict.   # tau --> the bandwidth.     
    # The values for which we are going to predict.
   # X_test includes nval evenly spaced values in the domain of X.
    X_test = np.linspace(-500, 700, nval) 
    
   # Empty list for storing predictions. 
    preds = [] 
    # Predicting for all nval values and storing them in preds. 
    for point in X_test: 
        theta, pred, err = predict(X, y, point, tau) 
        preds.append(pred)
        
   # Reshaping X_test and preds
    X_test = np.array(X_test).reshape(nval,1)
    preds = np.array(preds).reshape(nval,1)
    
   # Plotting 
    plt.plot(X, y, 'b.')
    plt.plot(X_test, preds, 'r.') # Predictions in red color.
    plt.show()
    x=3

X = df[['b']].to_numpy()
y = df[['y']].to_numpy()
plot_predictions(X, y, 0.08, 100)
