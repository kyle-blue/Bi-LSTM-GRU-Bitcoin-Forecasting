import tensorflow as tf
import tensorflow.python.keras.backend as K


def extreme_mae(y_true, y_pred):
    EXTREMES = 0.1 # 0.1 = 10% -- Work out upper 10% and lower 10% 

    count = len(y_pred)
    upper = int((1 - EXTREMES) * count)
    lower = int(EXTREMES * count)
    upper_bound = y_pred[upper]
    lower_bound = y_pred[lower]

    upper_mask = y_pred > upper_bound ## [true, false, false, ...] etc
    lower_mask = y_pred > lower_bound ## [true, false, false, ...] etc
    mask = tf.logical_or(upper_mask, lower_mask) # Combine Masks

    pred_slice = tf.boolean_mask(y_pred, mask)
    true_slice = tf.boolean_mask(y_true, mask)

    return K.mean(tf.abs(true_slice - pred_slice))