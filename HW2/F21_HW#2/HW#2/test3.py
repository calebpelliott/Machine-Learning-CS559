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

df = pd.read_csv("C:/Users/caleb/Documents/git/Machine-Learning-CS559/HW2/F21_HW#2/HW#2/heart.csv")

from sklearn import datasets

clf2 = Perceptron(eta0=0.5, random_state=1,max_iter=1000)

X = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']]
y = df[['output']]
X_heart = X
y_heart = y.values.ravel()
clf2.fit(X,y)
x = clf2.coef_
y_pred = clf2.predict(X)
clf2_accuracy = accuracy_score(y,y_pred)
clf2_accuracy

def plot_decision_boundary(X, theta):
    
    # X --> Inputs
    # theta --> parameters
    
    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -theta[1]/theta[2]
    c = -theta[0]/theta[2]
    x2 = m*x1 + c
    
    # Plotting
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "r^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Perceptron Algorithm')
    plt.plot(x1, x2, 'y-')
    plt.show()

def step_func(z):
        return 1.0 if (z > 0) else 0.0

def perceptron(X, y, lr, epochs,X_df):
    
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.
    
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    X_df.insert(0, 'bias', [1]*m)
    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n+1,1))
    
    # Empty list to store how many examples were 
    # misclassified at every iteration.
    n_miss_list = []
    
    # Training.
    for epoch in range(epochs):
        
        # variable to store #misclassified.
        n_miss = 0
        
        # looping for every example.
        #for i in X.iterrows():
        #    row_i = i[1].to_numpy()
            
        for idx, x_i in enumerate(X):
            #x_ii = i[1].values
            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1,1)
            
            yy = np.dot(x_i.T, theta)
            yyy = x_i.T.dot(theta)
            # Calculating prediction/hypothesis.
            y_hat = step_func(np.dot(x_i.T, theta))
            yyh = np.squeeze(y_hat)
            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[idx]) != 0:
                
                n_wrong = 0
                r_i = 0
                for row in X_df.to_numpy():
                    row = row.reshape(-1,1)
                    g = row.T.dot(theta)
                    guess = 0 if g<=0 else 1
                    if y[r_i] != guess:
                        n_wrong += 1
                    r_i += 1
                theta += lr*((y[idx] - y_hat)*x_i)
                
                # Incrementing by 1.
                n_miss += 1
        n_wrong = 0
        r_i = 0
        for row in X_df.to_numpy():
            row = row.reshape(-1,1)
            g = row.T.dot(theta)
            guess = 0 if g<=0 else 1
            if y[r_i] != guess:
                n_wrong += 1
            r_i += 1
    #print(pred.head(5))
        # Appending number of misclassified examples
        # at every iteration.
        n_miss_list.append(n_miss)
        if len(n_miss_list) == 99:
            x=3
    
        
    #print(pred.head(5))
    pred2 = [0 if x.T.dot(theta)<=0 else 1 for x in X_df.to_numpy()]
    return theta, n_miss_list

X, y = datasets.make_blobs(n_samples=150,n_features=2,
                           centers=2,cluster_std=1.05,
                           random_state=2)
fig = plt.figure(figsize=(10,8))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')
#plt.show()
theta, miss_l = perceptron(X_heart.to_numpy(), y_heart, 0.5, 101, X_heart)
#plot_decision_boundary(X, theta)

#t, n = perceptron(X,y,.5,1000)
z = 4

def my_Perceptron(df, eta, max_iter):
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]

    df_np = df.to_numpy()
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    num_features = X_np.shape[1]
    num_observations = X_np.shape[0]

    feature_weights = np.zeros((num_features + 1, 1))

    #Learning
    for iter in range(max_iter):
        row_i = 0
        df_shuffle = np.copy(df_np)
        np.random.shuffle(df_shuffle)
        for row in df_shuffle:
            y_actual = row[-1]
            #Insert 1 at front
            row = np.insert(row, 0, 1)
            #Remove y at end
            row = row[:-1]
            row = row.reshape(-1,1)

            intm = row.T.dot(feature_weights)
            guess = 0 if intm<=0 else 1
            if(guess != y_actual):
               #Positively bias
                if not guess:
                    feature_weights += eta*row
                #Negatively bias
                else:
                    feature_weights -= eta*row
            row_i += 1

    #Predicting
    X.insert(0, 'bias', [1]*num_observations)
    pred = X.dot(feature_weights).to_numpy()
    pred = [0 if x<=0 else 1 for x in pred]
    return pred
y_guess = my_Perceptron(df, .1, 1000)
acc = accuracy_score(y_heart,np.array(y_guess))
x=3