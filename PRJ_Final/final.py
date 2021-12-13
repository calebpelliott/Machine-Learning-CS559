# example of loading the mnist dataset
from hashlib import new
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier, plot_tree
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
#plt.show()

my_images = []
for i in range(10):
	for n in range(1,6):
		img = Image.open(f"./PRJ_Final/my_digits/{i}/{i}_{n}.png").convert('L')
		arr = np.array(img.getdata(), dtype=np.uint8)
		new_img = np.zeros(shape=(28,28))

		for p in range(28):
			slice = arr[p*28 : p*28+28]
			new_img[p] = slice
		my_images.append(new_img)


my_digitsX = np.array(my_images)
my_digitsX[0] = new_img
my_digitsY = np.array([0]*5+[1]*5+[2]*5+[3]*5+[4]*5+[5]*5+[6]*5+[7]*5+[8]*5+[9]*5, dtype=np.uint8)

lr_model = linear_model.LogisticRegression()
nsamples, nx, ny = trainX.shape
X_d2_train_dataset = trainX.reshape((nsamples,nx*ny))

nsamples, nx, ny = testX.shape
X_d2_test_dataset = testX.reshape((nsamples,nx*ny))

lr_model.fit(X_d2_train_dataset, trainy)
print(lr_model.score(X_d2_train_dataset, trainy))
print(lr_model.score(X_d2_test_dataset, testy))

from sklearn import svm

svm_model = svm.LinearSVC()

svm_model.fit(X_d2_train_dataset,trainy)
print(svm_model.score(X_d2_train_dataset, trainy))
print(svm_model.score(X_d2_test_dataset, testy))

xgb = XGBClassifier(objective='multiclass:softmax', 
	n_estimators = 325,
	learning_rate = 0.1,
	max_depth = 1
	)

xgb.fit(X_d2_train_dataset,trainy)
predict = xgb.predict(X_d2_train_dataset)
print(sum(predict == trainy)/len(trainy))
predict = xgb.predict(X_d2_test_dataset)
print(sum(predict == testy)/len(testy))
