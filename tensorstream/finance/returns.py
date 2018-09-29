import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Return(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def properties(self, value):
    shape = self.concat([self.period], value.shape)
    return value, (
      tf.constant(0),
      tf.zeros(shape, dtype=value.dtype)
    )

  def step(self, value, iteration, prev_buffer):
    prev_value = prev_buffer[self.period - 1]
    next_buffer = roll(value, prev_buffer)

    return_ = tf.cond(
      tf.logical_or(
        iteration < self.period - 1,
        tf.equal(prev_value, 0)
      ),
      lambda: tf.zeros(tf.shape(value), dtype=value.dtype),
      lambda: tf.divide(value, prev_value) - tf.constant(1.0)
    )
    return return_, (iteration + 1, next_buffer)

class LogarithmicReturn(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def properties(self, value):
    shape = self.concat([self.period], value.shape)
    return value, (
      tf.constant(0),
      tf.zeros(shape, dtype=value.dtype)
    )

  def step(self, value, iteration, prev_buffer):
    prev_value = prev_buffer[self.period - 1]
    next_buffer = roll(value, prev_buffer)

    return_ = tf.cond(
      tf.logical_or(
        iteration < self.period - 1,
        tf.equal(prev_value, 0)
      ),
      lambda: tf.zeros(tf.shape(value), dtype=value.dtype),
      lambda: tf.log(tf.divide(value, prev_value))
    )
    return return_, (iteration + 1, next_buffer)
