import tensorflow as tf

# RPS function implemented with tensorflow.
# This can be used directly as a loss function.
def rps(y, y_hat):
    c_y_hat = tf.math.cumsum(y_hat, axis=-1)
    c_y = tf.math.cumsum(y, axis=-1)
    return tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(c_y_hat - c_y), axis=-1))
