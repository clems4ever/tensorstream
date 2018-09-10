import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class MovingStandardDeviation(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def dtype(self, value):
    return value.dtype

  def shape(self, value):
    return tf.shape(value)

  def initial_state(self, value):
    shape = self.concat([self.period], tf.shape(value))
    return (
      tf.zeros(shape, dtype=value.dtype),
      tf.constant(0)
    )

  def step(self, value, previous_last_values, iteration):
    next_last_values = roll(value, previous_last_values)

    def compute_volatility():
      _, var = tf.nn.moments(next_last_values, axes=[0])
      return tf.sqrt(var)

    shape = self.shape(value)
    volatility = tf.cond(
      iteration < self.period - 1,
      lambda: tf.zeros(shape, dtype=value.dtype),
      compute_volatility
    )

    return volatility, (next_last_values, iteration + 1)
