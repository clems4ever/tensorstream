import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Skewness(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def initial_state(self, return_):
    shape = self.concat([self.period], tf.shape(return_))
    return (tf.zeros(shape, dtype=return_.dtype), tf.constant(0))

  def step(self, return_,
    last_returns, iteration):
    def compute_skewness(values):
      mean_x, var_x = tf.nn.moments(values, axes=0)
      stddev_x = tf.sqrt(var_x)
      mean_x3 = tf.reduce_mean(tf.pow(values, 3))
      # Formula available here: https://en.wikipedia.org/wiki/Skewness
      skewness = (mean_x3 - 3 * mean_x * tf.pow(stddev_x, 2) - tf.pow(mean_x, 3)) / tf.pow(stddev_x, 3)
      return tf.cond(
        tf.equal(stddev_x, 0.0),
        lambda: 0.0,
        lambda: skewness
      )
    
    new_returns = roll(return_, last_returns)
    skewness = tf.cond(
      tf.less(iteration, self.period),
      lambda: compute_skewness(new_returns[0:iteration + 1]),
      lambda: compute_skewness(new_returns))

    return skewness, (new_returns, iteration + 1)
