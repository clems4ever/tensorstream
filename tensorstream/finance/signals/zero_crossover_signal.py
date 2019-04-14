import tensorflow as tf

from tensorstream.streamable import Streamable


def sign(x):
    return tf.where(
        tf.greater_equal(tf.sign(x), tf.zeros(tf.shape(x))),
        tf.fill(tf.shape(x), 1.0),
        tf.fill(tf.shape(x), -1.0),
    )


class ZeroCrossoverSignal(Streamable):
    def __init__(self):
        super().__init__()

    def step(self, value, is_warmup=None, previous_direction=None):
        zeros = tf.zeros(tf.shape(value), dtype=value.dtype)

        if is_warmup is None:
            is_warmup = tf.constant(True)
        if previous_direction is None:
            previous_direction = zeros

        def warmup():
            return (zeros, sign(value))

        def nominal():
            is_zero_crossed = tf.logical_and(
                tf.less(sign(value * previous_direction), zeros),
                tf.logical_not(tf.equal(value, zeros)),
            )
            negative_direction = tf.negative(previous_direction)
            next_signal = tf.where(is_zero_crossed, negative_direction, zeros)
            next_direction = tf.where(
                is_zero_crossed, negative_direction, previous_direction
            )
            return next_signal, next_direction

        next_signal, next_direction = tf.cond(is_warmup, warmup, nominal)
        return (
            tf.cast(next_signal, tf.int32),
            (tf.constant(False), next_direction),
            (is_warmup, previous_direction),
        )
