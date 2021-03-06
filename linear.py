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

weights = tf.Variable(tf.truncated_normal([784, 10], stddev=0.05))
biases = tf.Variable(tf.ones([10]))
func = tf.matmul(X, weights) + biases
pred = tf.nn.softmax(func)  

# Implement cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))

# Minimize cross_entropy using gradient descent algorithm with learning rate
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_list=[] 
    test_list=[] 
    # Training 70 epochs
    for epoch in range(training_epochs+1):
	avg_train=0
	avg_test=0
	tot_batch=int(mnist.train.num_examples/batch_size)

        for bc in range(tot_batch):
	        batch_x, batch_y = mnist.train.next_batch(batch_size)
	        _,train_accuracy = sess.run([train_step,accuracy], 
			feed_dict={X: batch_x, Y: batch_y})
		avg_train += train_accuracy / tot_batch
	        test_accuracy = sess.run(accuracy, 
		        feed_dict={X: mnist.test.images, Y: mnist.test.labels})
		avg_test += test_accuracy / tot_batch
        print("Epoch: %d, Train accuracy: %.4f, Test accuracy: %.4f" 
            % (epoch+1, avg_train, avg_test))
        train_list.append(avg_train*100) 
        test_list.append(avg_test*100) 
       
	
    # Print confusion matrix
    cm_true = np.argmax(mnist.test.labels, 1)
    cm_pred = sess.run(tf.argmax(pred, 1), 
		feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    cm = confusion_matrix(y_true = cm_true, y_pred = cm_pred)
    print(cm)

    # Plot accuracy
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axis([0, 70, 0, 100])
    plt.plot(np.array(range(0,training_epochs+1)), np.array(train_list), 'r-', label='train')
    plt.plot(np.array(range(0,training_epochs+1)), np.array(test_list), 'k-', label='test')
    plt.show()
    plt.savefig('plot1.png')

