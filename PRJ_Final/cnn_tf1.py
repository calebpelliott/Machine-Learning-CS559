from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import math
import argparse

import tensorflow as tf
import tensorflow_datasets
#mnist = tensorflow_datasets.load('mnist')
from sklearn import datasets
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing

data = datasets.load_digits()

def get_weights(shape):
    data = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(data)

def get_biases(shape):
    data = tf.constant(0.1, shape=shape)
    return tf.Variable(data)

def create_layer(shape):
    # Get the weights and biases
    W = get_weights(shape)
    b = get_biases([shape[-1]])

    return W, b

def convolution_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
            padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

def CNN(X_train, Y_train, X_test, Y_test, my_digitsX, my_digitsY):
    
    # The images are 28x28, so create the input layer 
    # with 784 neurons (28x28=784) 
    x = tf.placeholder(tf.float32, [None, 784])

    # Reshape 'x' into a 4D tensor 
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Define the first convolutional layer
    W_conv1, b_conv1 = create_layer([5, 5, 1, 32])

    # Convolve the image with weight tensor, add the 
    # bias, and then apply the ReLU function
    h_conv1 = tf.nn.relu(convolution_2d(x_image, W_conv1) + b_conv1)

    # Apply the max pooling operator
    h_pool1 = max_pooling(h_conv1)

    # Define the second convolutional layer
    W_conv2, b_conv2 = create_layer([5,5,32,64])

    # Convolve the output of previous layer with the 
    # weight tensor, add the bias, and then apply 
    # the ReLU function
    h_conv2 = tf.nn.relu(convolution_2d(h_pool1,W_conv2)+ b_conv2)

    # Apply the max pooling operator
    h_pool2 = max_pooling(h_conv2)

    # Define the fully connected layer
    W_fc1, b_fc1 = create_layer([3136, 1024])

    # Reshape the output of the previous layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3136])

    # Multiply the output of previous layer by the 
    # weight tensor, add the bias, and then apply 
    # the ReLU function
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Define the dropout layer using a probability placeholder
    # for all the neurons
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Define the readout layer (output layer)
    W_fc2, b_fc2 = create_layer([1024, 10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define the entropy loss and the optimizer
    y_loss = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_loss))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Define the accuracy computation
    predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_loss, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

    # Create and run a session
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)

    # Start training
    batch_size = 75
    num_iterations = math.ceil(X_train.shape[0] / batch_size)
    start = 0
    end = batch_size
    for i in range(num_iterations):
        batch = (X_train[start:end], Y_train[start:end])
        start = end
        end = min(end + batch_size, X_train.shape[0])

        # Print progress
        if i % 50 == 0:
            cur_accuracy = accuracy.eval(feed_dict = {
                    x: batch[0], y_loss: batch[1], keep_prob: 1.0})
            print('Iteration', i, ', Accuracy =', cur_accuracy)

        # Train on the current batch
        optimizer.run(feed_dict = {x: batch[0], y_loss: batch[1], keep_prob: 0.5})

    # Compute accuracy using test data
    test_accuracy = accuracy.eval(feed_dict = {
            x: X_test, y_loss: Y_test,
            keep_prob: 1.0})
    print('Test accuracy =', test_accuracy)

    # Compute accuracy using test data
    digits_accuracy = accuracy.eval(feed_dict = {
            x: my_digitsX, y_loss: my_digitsY,
            keep_prob: 1.0})
    print('Digits accuracy =', digits_accuracy)
    
    return [test_accuracy,digits_accuracy]


rootdir = './PRJ_Final/my_digits'

def read_img(img_path):
    img = cv2.imread(img_path)
    x_img = np.array([0] * 784)

    for i in range(28):
        for j in range(28):
            idx = i * 28 + j
            x_img[idx] = 255 - img[i][j][0]
            
    return x_img, img
from PIL import Image
my_images = []
for i in range(10):
	for n in range(1,6):
		img = Image.open(f"./PRJ_Final/my_digits/{i}/{i}_{n}.png").convert('L')
		arr = np.array(img.getdata(), dtype=np.uint8)
		new_img = np.zeros(shape=(28,28))

		for p in range(28):
			slice = arr[p*28 : p*28+28]
			new_img[p] = slice
		my_images.append(new_img.flatten())


ohe = preprocessing.OneHotEncoder()
my_digitsX = np.array(my_images)
my_digitsY = np.array([0]*5+[1]*5+[2]*5+[3]*5+[4]*5+[5]*5+[6]*5+[7]*5+[8]*5+[9]*5, dtype=np.uint8)
my_digitsY = my_digitsY.reshape(-1,1)
ohe.fit(my_digitsY)
my_digitsY = ohe.transform(my_digitsY).toarray()
my_digitsY = np.array(my_digitsY)


(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

ohe.fit(train_y)
train_y = ohe.transform(train_y).toarray()
train_y = np.array(train_y)

ohe.fit(test_y)
test_y = ohe.transform(test_y).toarray()
test_y = np.array(test_y)

newX = []
for i in train_X:
    newX.append(i.flatten())

newX_test = []
for i in test_X:
    newX_test.append(i.flatten())

newX_test = np.array(newX_test)
newX = np.array(newX)

CNN(newX, train_y, newX_test, test_y, my_digitsX, my_digitsY)