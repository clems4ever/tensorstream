import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SharpeRatio(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def initial_state(self, return_, risk_free_rate):
    shape = self.concat([self.period], tf.shape(return_))
    return tf.zeros(shape, return_.dtype)

  def step(self, return_, risk_free_rate, last_adjusted_values):
    def compute_sharpe_ratio(adjusted_values):
      mean, var = tf.nn.moments(adjusted_values, axes=0)
      stddev = tf.sqrt(var)
      return mean / stddev
    
    new_adjusted_values = roll(return_ - risk_free_rate, last_adjusted_values)
    sharpe_ratio = tf.cond(
      tf.reduce_any(tf.equal(new_adjusted_values, 0)),
      lambda: tf.zeros(tf.shape(return_), dtype=return_.dtype),
      lambda: compute_sharpe_ratio(new_adjusted_values))

    return sharpe_ratio, new_adjusted_values
