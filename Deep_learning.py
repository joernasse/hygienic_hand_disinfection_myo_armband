# # import TF.Learn
# import numpy
# import tensorflow as tf
# from keras.datasets import mnist
# from sklearn.model_selection import train_test_split
#
# from Classification import TEST_SIZE
# from Helper_functions import divide_chunks
#
#
# def dnn_default(x_data, label):
#     # init = tf.global_variables_initializer()
#     # saver = tf.train.Saver()
#
#     # Training section
#     n_epochs = 40
#     batch_size = 50
#
#     x_train, x_test, y_train, y_test = train_test_split(x_data, label, test_size=TEST_SIZE, random_state=42)
#     (x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()
#
#     x_train = numpy.asarray(x_train)
#     y_train = numpy.asarray(y_train)
#     x_test = numpy.asarray(x_test)
#     y_test = numpy.asarray(y_test)
#
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(8, 50)),
#         tf.keras.layers.Dense(128, activation='elu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(13, activation='softmax')
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     print("Training")
#     model.fit(x_train, y_train, epochs=50)
#
#     print("Evaluate")
#     model.evaluate(x_test, y_test)
#
#
#     print("x")
#
#
