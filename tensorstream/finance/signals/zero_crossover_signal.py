import tensorflow as tf

from tensorstream.streamable import Streamable

class ZeroCrossoverSignal(Streamable):
  def __init__(self):
    super().__init__()

  def initial_state(self, value):
    return (
      tf.constant(True),
      tf.zeros(tf.shape(value), dtype=value.dtype)
    )

  def step(self, value, is_warmup, last_value):
    zeros_i = tf.zeros(tf.shape(value), dtype=tf.int32)
    zeros = tf.zeros(tf.shape(value), dtype=value.dtype)

    def warmup():
      return zeros_i

    def nominal():
      is_zero_crossed = tf.less(tf.sign(value * last_value), zeros)
      set_signal = tf.to_int32(tf.sign(value - last_value))
      return tf.where(
        tf.logical_and(
          is_zero_crossed,
          tf.logical_not(tf.equal(last_value, 0))
        ),
        set_signal,
        zeros_i
      )

    new_signal = tf.cond(is_warmup, warmup, nominal)
    return new_signal, (tf.constant(False), value)
