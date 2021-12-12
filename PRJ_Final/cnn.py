import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
#import tensorflow_datasets
#mnist = tensorflow_datasets.load('mnist')
#import input_data
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers.core import Dense, Dropout

class CNN_class(Model):
    def __init__(self):
        super(CNN_class, self).__init__()
        # Define the first convolutional layer
        #W_conv1, b_conv1 = create_layer([5, 5, 1, 32])
        # Convolve the image with weight tensor, add the
        # bias, and then apply the ReLU function
        #h_conv1 = tf.nn.relu(convolution_2d(x_image, W_conv1) + b_conv1)
        self.conv1 = layers.Conv2D(32, 5, activation=tf.nn.relu)

        # Apply the max pooling operator
        #h_pool1 = max_pooling(h_conv1)
        self.h_pool1 = layers.MaxPool2D(2,2)

        # Define the second convolutional layer
        #W_conv2, b_conv2 = create_layer([5, 5, 32, 64])

        # Convolve the output of previous layer with the
        # weight tensor, add the bias, and then apply
        # the ReLU function
        #h_conv2 = tf.nn.relu(convolution_2d(h_pool1, W_conv2) + b_conv2)
        self.h_conv2 = layers.Conv2D(64, 3, activation=tf.nn.relu)

        # Apply the max pooling operator
        #h_pool2 = max_pooling(h_conv2)
        self.h_pool2 = layers.MaxPool2D(2,2)
        # Define the fully connected layer
        #W_fc1, b_fc1 = create_layer([7 * 7 * 64, 1024])
        self.fc1 = layers.Dense(1024)
        self.flatten = layers.Flatten()
        self.dropout = Dropout(0.5)
        self.out = layers.Dense(10)

    def layer1(self,X):
        X = self.conv1(X)
        X = self.h_pool1(X)
        return X

    def layer2(self,X):
        X = self.h_conv2(X)
        X = self.h_pool2(X)
        return X
    def getOutputLayer(self,X):
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.dropout(X, True)
        X = self.out(X)
        return X
    def runNetwork(self,X):
        X = tf.reshape(X, [-1, 28, 28, 1])
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.getOutputLayer(X)
        return X
    def predict(self,X):
        return tf.nn.softmax(self.runNetwork(X))
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    predict = tf.argmax(y_pred, 1)
    predict = np.array(predict)
    actual = np.array(y_true)
    acc = (actual == predict).sum()/len(actual)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    print(f"{acc}")
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
def CNN(X_train, Y_train, X_test, Y_test):
    
    # The images are 28x28. Create the input layer
    #x = tf.placeholder(tf.float32, [None, 784])
    ## UPDATING FOR TENSORFLOW2
    
    conv2 = layers.Conv2D(64, 3)
    pool1 = layers.MaxPool2D(2,2)
    pool2 = layers.MaxPool2D(2,2)
    flatten = layers.Flatten()
    fc1 = Dense(1024)
    dropout = Dropout(0.5)
    out = layers.Dense(10)
    # Reshape 'x' into a 4D tensor
    #x_image = tf.reshape(x, [-1, 28, 28, 1])

    
    
    
    
    
    
    # Reshape the output of the previous layer
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # Multiply the output of previous layer by the
    # weight tensor, add the bias, and then apply
    # the ReLU function * Use "tf.matmul" for matrix multiplication
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Define the dropout layer using a probability placeholder
    # for all the neurons
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob )

    # Define the readout layer (output layer)
    #W_fc2, b_fc2 = create_layer([1024, 10])
    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define the entropy loss and the optimizer
    #y_loss = tf.placeholder(tf.float32, [None, 10])
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_loss))
    #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    optimizer = tf.optimizers.Adam(1e-4)
    # Define the accuracy computation
    #predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_loss, 1))
    #accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

    # Create and run a session
    #sess = tf.InteractiveSession()
    #init = tf.initialize_all_variables()
    #sess.run(init)
    CNN = CNN_class()
    # Start training
    batch_size = 75
    num_iterations = math.ceil(X_train.shape[0] / batch_size)
    start = 0
    end = batch_size
    print('\nTraining the model....')
    for i in range(num_iterations):
        #batch_X = X_train[start:end] 
        #batch_X = np.reshape(batch_X, (-1, 28, 28, 1))
        #batch_y = Y_train[start:end]
        #batch_y = np.reshape(batch_y, (-1, 28, 28, 1))
        batch = (X_train[start:end], Y_train[start:end])
        batch_x = tf.convert_to_tensor(batch[0])
        batch_y = tf.convert_to_tensor(batch[1])
        batch_y = tf.cast(batch_y, tf.int64)
        
        with tf.GradientTape() as gradient:
            results = CNN.runNetwork(batch_x)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=results, labels=batch_y))
        tv = CNN.trainable_variables
        opt_params = gradient.gradient(loss, tv)
        changes = zip(opt_params, tv)
        optimizer.apply_gradients(changes)

        # Print progress
        if i % 50 == 0:
            prediction_y = CNN.predict(batch_x)
            a = accuracy(prediction_y, batch_y)
            print(f"prediction: {prediction_y}, actual: {batch_y}")
            print(f"Step: {i}, accuracy: {a}")
            x=3
    # Compute accuracy using test data
    test_accuracy = accuracy.eval(feed_dict = {
            x: X_test, y_loss: Y_test,
            keep_prob: 1.0})
    print('Test accuracy =', test_accuracy)
    
    return test_accuracy

if __name__ == '__main__':
    x = 3

    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
    train_examples = None
    train_labels = None
    test_examples = None
    test_labels = None
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']
    new_train_X = np.zeros(shape=(60000,784), dtype=np.float32)
    new_train_y = np.zeros(shape=(60000,10), dtype=np.uint8)
    new_test_X = np.zeros(shape=(10000,784), dtype=np.float32)
    new_test_y = np.zeros(shape=(10000,10), dtype=np.uint8)
    for img_set in [[train_examples, new_train_X],[train_labels, new_train_y],[test_examples, new_test_X],[test_labels,new_test_y]]:
        for img_num in range(len(img_set[0])):
            img_set[1][img_num] = img_set[0][img_num].flatten()

    (trainX, trainy), (testX, testy) = mnist.load_data()
    trainX = np.array(trainX, np.float32)
    testX = np.array(testX, np.float32)
    train_y = np.array(trainy, np.int64)
    #trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    #testX = testX.reshape((testX.shape[0], 28, 28, 1))
    CNN(trainX/255.0, trainy, testX/255.0, testy)
    CNN(new_train_X/255.0, new_train_y, new_test_X/255.0, new_test_y)