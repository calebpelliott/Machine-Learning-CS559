from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.utils.extmath import weighted_mode
np.set_printoptions(precision=4)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

def my_LogisticRegression(df, eta, max_iter):
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]

    df_np = df.to_numpy()
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    num_features = X_np.shape[1]
    num_observations = X_np.shape[0]

    feature_weights = np.zeros((num_features, 1))
    bias = 0
    #Learning
    for iter in range(max_iter):
        for i in range(num_observations):
            start = i*num_observations
            end = start + num_observations
            
            x_samp = X[start:end]
            y_samp = y[start:end]

            pred = x_samp.dot(feature_weights) + bias
            #sigmoid
            pred = float(1)/(1 + np.exp(-pred))

            #Partials
            y_diff = pred - y_samp
            differ_w = (1/num_observations) * x_samp.T.dot(y_diff)
            differ_b = (1/num_observations) * np.sum(y_diff)

            feature_weights = feature_weights - (eta*differ_w)
            bias = bias - (eta*differ_b)

    #Predicting
    pred = X.dot(feature_weights) + bias
    pred = pred = float(1)/(1 + np.exp(-pred))
    pred = [0 if x <= .5 else 1 for x in pred]