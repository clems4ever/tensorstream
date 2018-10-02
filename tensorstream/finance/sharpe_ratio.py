import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SharpeRatio(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def step(self, return_, risk_free_rate,
    last_adjusted_values=None, iteration=None):
    if iteration is None:
      iteration = tf.constant(0)
    if last_adjusted_values is None:
      shape = self.concat([self.period], tf.shape(return_))
      last_adjusted_values = tf.zeros(shape, return_.dtype)

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

    return sharpe_ratio, (new_adjusted_values, iteration + 1), (last_adjusted_values, iteration)
