import math
import tensorflow as tf

from tensorstream.streamable import Streamable

class ZeroCrossoverSignal(Streamable):
  def __init__(self, dtype=tf.float32, shape=()):
    super().__init__(tf.int32, shape)
    self.input_dtype = dtype
    self.initial_state = (tf.constant(True), tf.fill(shape, math.nan))

  def step(self, value, is_warmup, last_value):
    zeros_i = tf.zeros(self.shape, dtype=tf.int32)
    zeros = tf.zeros(self.shape, dtype=self.input_dtype)

    def warmup():
      return zeros_i

    def nominal():
      is_zero_crossed = tf.less(tf.sign(value * last_value), zeros)
      set_signal = tf.to_int32(tf.sign(value - last_value))
      return tf.where(is_zero_crossed, set_signal, zeros_i)

    new_signal = tf.cond(is_warmup, warmup, nominal)
    return new_signal, (tf.constant(False), value)
