import numpy as np
import tensorflow as tf
import math
from batch_norm import batch_norm as batno

tf.set_random_seed(777)

X_data = np.load("ADL_Fall_X.npy")
Y_data = np.load("ADL_Fall_Y.npy")

shuffle = np.arange(Y_data.shape[0])
np.random.shuffle(shuffle)
X_data = X_data[shuffle]
Y_data = Y_data[shuffle]
print(X_data.shape, Y_data.shape)

batch_size = 100
num_epochs = 10
num_iterations = int(math.ceil(X_data.shape[0] / batch_size))

X = tf.placeholder(tf.float32, [None, 3, 582])
X_con = tf.reshape(X, [-1, 3, 582, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([1, 10, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_con, W1, strides=[1, 1, 2, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 1, 4, 1],
                    strides=[1, 1, 4, 1], padding='SAME')
print(L1.shape)

W2 = tf.Variable(tf.random_normal([1, 5, 32, 32], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 2, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 1, 3, 1],
                    strides=[1, 1, 3, 1], padding='SAME')
print(L2.shape)
L2_flat = tf.reshape(L2, [-1, 3 * 13 * 32])

W3 = tf.get_variable("W3", shape=[3 * 13 * 32, 1000],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1000]))
bn3 = batno(tf.matmul(L2_flat, W3) + b3)
# L3 = tf.nn.relu(bn3)
L3 = tf.sigmoid(bn3)

W4 = tf.get_variable("W4", shape=[1000, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([100]))
bn4 = batno(tf.matmul(L3, W4) + b4)
# L4 = tf.nn.relu(bn4)
L4 = tf.sigmoid(bn4)

W5 = tf.get_variable("W5", shape=[100, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(L4, W5) + b5)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0
        avg_accuracy = 0

        for iteration in range(num_iterations):
            batch_xs = X_data[batch_size * iteration: batch_size * (iteration + 1), :, :]
            batch_ys = Y_data[batch_size * iteration: batch_size * (iteration + 1), :]
            _, cost_val, acc_val = sess.run([train, cost, accuracy],
                                            feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations
            avg_accuracy += acc_val / num_iterations

        print(f"Epoch: {(epoch + 1):04d}, Cost: {avg_cost:.9f}, Accuracy: {avg_accuracy:.2%}")

    acc = sess.run(accuracy, feed_dict={X: X_data, Y: Y_data})
    print(f"Accuracy: {(acc * 100):2.2f}%")

    show_X_data = X_data[-1:, :]
    show_Y_data = Y_data[-1:, :]

    pred = sess.run(predicted, feed_dict={X: show_X_data})
    for p, y in zip(pred, show_Y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    print(type(predicted))