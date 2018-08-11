import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Skewness(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = tf.fill((period,) + self.shape, math.nan)

  def step(self, return_, last_returns):
    def compute_skewness(values):
      mean, var = tf.nn.moments(values, axes=0)
      stddev = tf.sqrt(var)
      return tf.pow(mean, 3) / tf.pow(stddev, 3)
    
    new_returns = roll(return_, last_returns)
    skewness = tf.cond(tf.reduce_any(tf.is_nan(new_returns)),
      lambda: tf.fill(self.shape, math.nan),
      lambda: compute_skewness(new_returns))

    return skewness, new_returns
