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
from sklearn.datasets import make_classification 
df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/HW2/F21_HW#2/HW#2/heart.csv")

def my_LogisticRegression(df, eta, max_iter):
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]

    df_np = df.to_numpy()
    X_np = X.to_numpy()
    y_np = y.to_numpy().reshape(-1,1)#.reshape(-1,1)
    num_features = X_np.shape[1]
    num_observations = X_np.shape[0]

    feature_weights = np.zeros((num_features, 1))
    bias = 0
    sample_size = 303
    
    #Learning
    for iter in range(max_iter):
        reached_end = False
        index = 0
        while not reached_end:
            end_idx = index + sample_size
            if end_idx >= num_observations:
                end_idx = num_observations
                reached_end = True
            
            x_samp = X_np[index:end_idx]
            y_samp = y_np[index:end_idx]

            pred = x_samp.dot(feature_weights) + bias
            #sigmoid
            pred = float(1)/(1 + np.exp(-pred))

            #Partials
            y_diff = pred - y_samp
            differ_w = (1/num_observations) * x_samp.T.dot(y_diff)
            differ_b = (1/num_observations) * np.sum(y_diff)

            feature_weights = feature_weights - (eta*differ_w)
            bias = bias - (eta*differ_b)
            index += sample_size

    #Predicting
    pred = (X.dot(feature_weights) + bias).to_numpy()
    pred = pred = float(1)/(1 + np.exp(-pred))
    pred = [0 if x <= .5 else 1 for x in pred]
    return np.array(pred)

X = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']]
y = df[['output']]
new_df = df[['sex', 'cp', 'exng', 'slp', 'caa', 'thall', 'output']]
y_pred = my_LogisticRegression(new_df, .01, 1000)
y_heart = y.values.ravel()
## REMOVE ME
X,y = make_classification(n_features=2, n_redundant=0, 
                        n_informative=2, random_state=1, 
                        n_clusters_per_class=1)
c = accuracy_score(y_heart, y_pred)
xx=4