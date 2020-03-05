import numpy as np
import tensorflow as tf
import math

def make_model(model_path, sess, inputs, outputs):
    # inputs와 outputs가 여러 요소를 갖는다면 다중 입출력이 된다.
    converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
    flat_data = converter.convert()

    with open(model_path, 'wb') as f:
        f.write(flat_data)

tf.set_random_seed(777)

X_data = np.load("Ten_Move_X.npy")
Y_data = np.load("Ten_Move_Y.npy")

shuffle = np.arange(Y_data.shape[0])
np.random.shuffle(shuffle)
X_data = X_data[shuffle]
Y_data = Y_data[shuffle]
# print(X_data.shape, Y_data.shape)

num_movement = 10
batch_size = 100
num_epochs = 50
num_iterations = int(math.ceil(X_data.shape[0] / batch_size))

X = tf.placeholder(tf.float32, [None, 3, 582])
X_con = tf.reshape(X, [-1, 3, 582, 1])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, num_movement)
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_movement])

W1 = tf.Variable(tf.random_normal([1, 10, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_con, W1, strides=[1, 1, 2, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 1, 4, 1],
                    strides=[1, 1, 4, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([1, 5, 32, 32], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 2, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 1, 3, 1],
                    strides=[1, 1, 3, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 3 * 13 * 32])

W3 = tf.get_variable("W3", shape=[3 * 13 * 32, num_movement],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([num_movement]))
hypothesis = tf.matmul(L2_flat, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.argmax(hypothesis, axis=1)
predicted = tf.cast(prediction, dtype=tf.int32)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

    #     print(f"Epoch: {(epoch + 1):04d}, Cost: {avg_cost:.9f}, Accuracy: {avg_accuracy:.2%}")
    #
    # acc = sess.run(accuracy, feed_dict={X: X_data, Y: Y_data})
    # print(f"Accuracy: {(acc * 100):2.2f}%")
    #
    # show_X_data = X_data[-100:, :]
    # show_Y_data = Y_data[-100:, :]
    #
    # pred = sess.run(prediction, feed_dict={X: show_X_data})
    # for p, y in zip(pred, show_Y_data.flatten()):
    #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    make_model('Ten_Move_model.tflite', sess, [X], [predicted])