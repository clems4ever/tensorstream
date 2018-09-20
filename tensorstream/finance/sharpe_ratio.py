import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SharpeRatio(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def initial_state(self, return_, risk_free_rate):
    shape = self.concat([self.period], tf.shape(return_))
    return (tf.zeros(shape, return_.dtype), tf.constant(0))

  def step(self, return_, risk_free_rate,
    last_adjusted_values, iteration):
    def compute_sharpe_ratio(adjusted_values):
      mean, var = tf.nn.moments(adjusted_values, axes=0)
      stddev = tf.sqrt(var)
      return tf.cond(
        tf.equal(stddev, 0.0),
        lambda: 0.0,
        lambda: mean / stddev
      )
    
    new_adjusted_values = roll(return_ - risk_free_rate, last_adjusted_values)
    sharpe_ratio = tf.cond(
      tf.less(iteration, self.period),
      lambda: compute_sharpe_ratio(new_adjusted_values[0:iteration + 1]),
      lambda: compute_sharpe_ratio(new_adjusted_values))

    return sharpe_ratio, (new_adjusted_values, iteration + 1)
