import math
import tensorflow as tf

from tensorstream.operators import Streamable, shift

class GlobalMinimum(Streamable):
  def __init__(self, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.initial_state = tf.constant(math.inf)

  def step(self, value, global_min):
    new_min = tf.cond(tf.less(value, global_min),
      lambda: value, lambda: global_min)
    return new_min, new_min

class GlobalMaximum(Streamable):
  def __init__(self, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.initial_state = tf.constant(-math.inf)

  def step(self, value, global_max):
    new_max = tf.cond(tf.greater(value, global_max),
      lambda: value, lambda: global_max)
    return new_max, new_max

class LocalMinimum(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = tf.fill([period], math.inf)

  def step(self, value, last_values):
    new_last_values = shift(value, last_values)
    min_value = tf.reduce_min(new_last_values)
    return min_value, new_last_values

class LocalMaximum(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = tf.fill([period], -math.inf)

  def step(self, value, last_values):
    new_last_values = shift(value, last_values)
    max_value = tf.reduce_max(new_last_values)
    return max_value, new_last_values
