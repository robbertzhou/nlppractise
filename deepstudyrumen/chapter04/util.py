import tensorflow as tf

train, test = tf.keras.datasets.mnist.load_data()
print(train.shape)