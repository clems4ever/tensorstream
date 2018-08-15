import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class MovingStandardDeviation(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = tf.fill((period,) + shape, math.nan)

  def step(self, value, previous_last_values):
    next_last_values = roll(value, previous_last_values)

    def compute_volatility():
      _, var = tf.nn.moments(next_last_values, axes=[0])
      return tf.sqrt(var)

    has_nan = tf.reduce_any(tf.is_nan(next_last_values))
    volatility = tf.cond(has_nan,
      lambda: tf.fill(self.shape, math.nan),
      compute_volatility)

    return volatility, next_last_values

# Alias
Volatility = MovingStandardDeviation
