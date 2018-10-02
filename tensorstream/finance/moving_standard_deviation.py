import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class MovingStandardDeviation(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def step(self, value, iteration=None, prev_buffer=None):
    if iteration is None:
      iteration = tf.constant(0)
    if prev_buffer is None:
      shape = self.concat([self.period], tf.shape(value))
      prev_buffer = tf.zeros(shape, value.dtype)

    next_buffer = roll(value, prev_buffer)

    _, var = tf.nn.moments(next_buffer, axes=[0])
    sqrt_var = tf.sqrt(var)

    volatility = tf.cond(
      iteration < self.period - 1,
      lambda: tf.zeros(tf.shape(value), dtype=value.dtype),
      lambda: sqrt_var
    )
    return volatility, (iteration + 1, next_buffer), (iteration, prev_buffer)
