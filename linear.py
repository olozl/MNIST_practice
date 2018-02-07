from __future__ import print_function

# Download MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataset/", one_hot=True)

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Parameters
learning_rate = 0.001
training_epochs = 70
batch_size = 100
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

func = tf.matmul(X, weights) + biases
pred = tf.nn.softmax(func)  

# Implement cross entropy function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

# Minimize cross_entropy using gradient descent algorithm with learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initialize variables
init = tf.global_variables_initializer()

'''
plt.imshow(mnist.train.images[0].reshape((28,28)), cmap='gray')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('plot1.png')

plt.ion()
'''

with tf.Session() as sess:
    sess.run(init)

    # Training 70 epochs
    for epoch in range(training_epochs):
        for i in range(10000):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, train_accuracy = sess.run([train_step, accuracy],
		feed_dict={X: batch_x, Y: batch_y})
        test_accuracy = sess.run(accuracy, 
	    feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print("Epoch: %d, Train accuracy: %.4f, Test accuracy: %.4f" 
            % (epoch+1, train_accuracy, test_accuracy))
	
    # Print confusion matrix
    cm_true = np.argmax(mnist.test.labels, 1)
    cm_pred = sess.run(tf.argmax(pred, 1), 
		feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    cm = confusion_matrix(y_true = cm_true, y_pred = cm_pred)
    print(cm)

'''
plt.savefig('plot2.png')
plt.ioff()
'''
