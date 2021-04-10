import tensorflow as tf

def RSquaredMetric(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return R_squared

def RSquaredMetricNeg(y_true, y_pred):
    return -RSquaredMetric(y_true, y_pred)
