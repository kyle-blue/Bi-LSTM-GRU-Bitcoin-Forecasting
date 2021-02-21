import tensorflow as tf
import tensorflow.python.keras.backend as K
import sys


def extreme_mae(y_true, y_pred):
    EXTREMES = tf.constant(0.1) # 0.1 = 10% -- Work out upper 10% and lower 10% 
    
    count = tf.cast(len(y_pred), tf.dtypes.float32)
    if count < tf.constant(10.0):
        return 0.0
    upper = tf.cast(tf.constant(((tf.constant(1.0) - EXTREMES) * count)), tf.dtypes.int32)
    lower = tf.cast(tf.constant(EXTREMES * count), tf.dtypes.int32)
    tf.print("count:", count, "upper", upper, "lower", lower, output_stream=sys.stdout)
    upper_bound = y_pred[upper]
    lower_bound = y_pred[lower]

    upper_mask = y_pred > upper_bound ## [true, false, false, ...] etc
    lower_mask = y_pred > lower_bound ## [true, false, false, ...] etc
    mask = tf.logical_or(upper_mask, lower_mask) # Combine Masks

    pred_slice = tf.boolean_mask(y_pred, mask)
    true_slice = tf.boolean_mask(y_true, mask)

    return K.mean(tf.abs(true_slice - pred_slice))
    