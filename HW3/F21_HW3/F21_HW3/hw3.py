from typing import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('./F21_CS559_HW3_data.csv')
print(df.head(5))


plt.hist(df['c'])
#plt.show()

#Cut by mean
c_mean = df['c'].mean()
c_min = df['c'].min()
c_max = df['c'].max()
c_mid = (c_max + c_min) /2
                                       #c_mean
df['c'] = pd.cut(df['c'], bins=[c_min, c_mid, c_max], labels=[0,1])

true_count = Counter(list(df['class']))
#Calculate gini
as_list = list(df['c'])
count = Counter(as_list)
count_list = [count[0], count[1]]
occurences = list(Counter(list(df['c'])).values)
print(df)
x=3