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
      mean = tf.reduce_mean(next_last_values)
      squared_deviation = tf.square(next_last_values - tf.fill((self.period,) + self.shape, mean))
      mean_squared_deviation = tf.reduce_mean(squared_deviation) 
      return tf.sqrt(mean_squared_deviation)

    has_nan = tf.reduce_any(tf.is_nan(next_last_values))
    volatility = tf.cond(has_nan,
      lambda: tf.zeros(self.shape), 
      compute_volatility)

    return volatility, next_last_values

# Alias
Volatility = MovingStandardDeviation
