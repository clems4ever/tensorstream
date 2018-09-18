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
      mean, var = tf.nn.moments(values, axes=0)
      stddev = tf.sqrt(var)
      return tf.pow(mean, 3) / tf.pow(stddev, 3)
    
    new_returns = roll(return_, last_returns)
    skewness = tf.cond(
      tf.less(iteration, self.period),
      lambda: tf.zeros(tf.shape(return_), dtype=return_.dtype),
      lambda: compute_skewness(new_returns))

    return skewness, (new_returns, iteration + 1)
