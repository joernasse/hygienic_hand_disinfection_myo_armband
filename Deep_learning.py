# import TF.Learn
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Classification import TEST_SIZE
from Helper_functions import divide_chunks


def dnn_default(x_data, label):

    # DNN EMGIMU-default ->8*6+9*6=102
    n_inputs = 102
    n_hidden1 = 70
    n_hidden2 = 50
    n_output = 13
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_output, name="outputs")
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training section
    n_epochs = 40
    bach_size = 50

    X_train, X_test, y_train, y_test = train_test_split(x_data, label, test_size=TEST_SIZE, random_state=42)
    X_batch, y_batch = [], []  # batch berechnen!
    X_batch = divide_chunks(X_train, bach_size)
    y_batch = divide_chunks(y_train, bach_size)

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Genauigkeit Training", acc_train, "Genauigkeit Test", acc_test)
        save_path = saver.save(sess, "/dnn_model_final.ckpt")
    print("x")
