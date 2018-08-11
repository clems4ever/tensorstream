import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SharpeRatio(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = tf.fill((period,) + self.shape, math.nan)

  def step(self, return_, risk_free_rate, last_adjusted_values):
    def compute_sharpe_ratio(adjusted_values):
      mean, var = tf.nn.moments(adjusted_values, axes=0)
      stddev = tf.sqrt(var)
      return mean / stddev
    
    new_adjusted_values = roll(return_ - risk_free_rate, last_adjusted_values)
    sharpe_ratio = tf.cond(tf.reduce_any(tf.is_nan(new_adjusted_values)),
      lambda: tf.fill(self.shape, math.nan),
      lambda: compute_sharpe_ratio(new_adjusted_values))

    return sharpe_ratio, new_adjusted_values
