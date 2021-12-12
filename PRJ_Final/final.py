# example of loading the mnist dataset
from hashlib import new
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from PIL import Image
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


img = Image.open("./PRJ_Final/my_digits/3/3_2.png").convert('L')
arr = np.array(img.getdata(), dtype=np.uint8)
new_img = np.zeros(shape=(28,28))
for i in range(28):
	slice = arr[i*28 : i*28+28]
	new_img[i] = slice


plt.imshow(new_img, cmap=plt.get_cmap('gray'))
plt.show()

lr_model = linear_model.LogisticRegression()
nsamples, nx, ny = trainX.shape
X_d2_train_dataset = trainX.reshape((nsamples,nx*ny))

nsamples, nx, ny = testX.shape
X_d2_test_dataset = testX.reshape((nsamples,nx*ny))

lr_model.fit(X_d2_train_dataset, trainy)
print(lr_model.score(X_d2_train_dataset, trainy))
print(lr_model.score(X_d2_test_dataset, testy))


