#https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2
#https://towardsdatascience.com/perceptron-algorithm-in-python-f3ac89d2e537
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
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

clf1= LinearDiscriminantAnalysis()
columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
X = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']]
y = df[['output']]
sc = StandardScaler()
print(X)
#X = sc.fit_transform(X)
print(X)
clf1.fit_transform(X,y)
y_pred = clf1.predict(X)
print()
clf1_accuracy = accuracy_score(y['output'].values,y_pred)

# Assume target is last column
def my_LDA(df):
    X = df.iloc[:, 0:-1]
    y = df.iloc[:,-1]

    #Get target classes
    classes = y.unique()

    #Calculate feature means
    feature_means_per_class = pd.DataFrame(columns=classes)
    class_name = df.columns[-1]
    for i in df.groupby(class_name):
        class_i = i[0]
        class_rows = i[1].iloc[:,0:-1]
        feature_means_per_class[class_i] = class_rows.mean()

    #Calculate Sw
    num_features = X.shape[1]
    Sw = np.zeros((num_features, num_features))
    for i in df.groupby(class_name):
        class_i = i[0]
        class_rows = i[1]
        features = class_rows.iloc[:,0:-1]

        #Calculate Si
        Si = np.zeros((num_features, num_features))
        for j in features.iterrows():
            idx_j = j[0]
            row_j = j[1]
            x = row_j.values
            mean = feature_means_per_class[class_i].values
            x = x.reshape(-1,1)
            mean = mean.reshape(-1,1)

            #(x-m)(x-m)^T
            var = x - mean
            varT = var.T
            dot = var.dot(varT)
            Si = Si + dot

    #Find eigenvectors/values (W)
#heart df
my_LDA(df)

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)
print(X.head(5))


df = X.join(pd.Series(y, name='class'))
my_LDA(df)
print(df.head(5))
class_feature_means = pd.DataFrame(columns=wine.target_names)
for c, rows in df.groupby('class'):
    class_feature_means[c] = rows.mean()

print(class_feature_means)
d = df.groupby('class')
print(d.head(5))
within_class_scatter_matrix = np.zeros((13,13))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    print(rows)
    s = np.zeros((13,13))
    for index, row in rows.iterrows():
        mcc = class_feature_means[c].values.reshape(-1,1)
        x, mc = row.values.reshape(13,1), class_feature_means[c].values.reshape(13,1)
            
        s += (x - mc).dot((x - mc).T)
        
    within_class_scatter_matrix += s


feature_means = df.mean()
between_class_scatter_matrix = np.zeros((13,13))
for c in class_feature_means:    
    n = len(df.loc[df['class'] == c].index)
    
    mc, m = class_feature_means[c].values.reshape(13,1), feature_means.values.reshape(13,1)
    
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
for pair in pairs:
    print(pair[0])