import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Return(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = (0, tf.fill((self.period,) + shape, math.nan))

  def step(self, value, iteration, prev_buffer):
    prev_value = prev_buffer[self.period - 1]
    next_buffer = roll(value, prev_buffer)

    return_ = tf.cond(
      iteration < self.period - 1,
      lambda: tf.fill(self.shape, math.nan),
      lambda: tf.divide(value, prev_value) - tf.constant(1.0)
    )
    return return_, (iteration + 1, next_buffer)

class LogarithmicReturn(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = (0, tf.fill((self.period,) + shape, math.nan))

  def step(self, value, iteration, prev_buffer):
    prev_value = prev_buffer[self.period - 1]
    next_buffer = roll(value, prev_buffer)

    return_ = tf.cond(
      iteration < self.period - 1,
      lambda: tf.fill(self.shape, math.nan),
      lambda: tf.log(tf.divide(value, prev_value))
    )
    return return_, (iteration + 1, next_buffer)
