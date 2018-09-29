import tensorflow as tf

from tensorstream.common import shift
from tensorstream.streamable import Streamable

class GlobalMinimum(Streamable):
  def step(self, value, global_min=None,
    is_first_iteration=True):

    if global_min is None:
      global_min = tf.zeros(tf.shape(value), value.dtype)

    new_min = tf.cond(
      tf.logical_or(
        is_first_iteration,
        tf.less(value, global_min)
      ),
      lambda: value,
      lambda: global_min
    )
    return new_min, (new_min, False), (global_min, is_first_iteration)

class GlobalMaximum(Streamable):
  def step(self, value, global_max=None, is_first_iteration=True):
    if global_max is None:
      global_max = tf.zeros(tf.shape(value), value.dtype)

    new_max = tf.cond(
      tf.logical_or(
        is_first_iteration,
        tf.greater(value, global_max)
      ),
      lambda: value,
      lambda: global_max
    )
    return new_max, (new_max, False), (global_max, is_first_iteration)

class LocalExtremum(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

class LocalMinimum(LocalExtremum):
  def __init__(self, period):
    super().__init__(period)

  def step(self, value, last_values=None, iteration=0):
    if last_values is None:
      shape = self.concat([self.period], tf.shape(value))
      last_values = tf.zeros(shape, value.dtype)
    new_last_values = shift(value, last_values)
    min_value = tf.cond(
      tf.less(iteration, self.period - 1),
      lambda: tf.reduce_min(new_last_values[:iteration + 1]),
      lambda: tf.reduce_min(new_last_values)
    )
    return min_value, (new_last_values, iteration + 1), (last_values, iteration)

class LocalMaximum(LocalExtremum):
  def __init__(self, period):
    super().__init__(period)

  def step(self, value, last_values=None, iteration=0):
    if last_values is None:
      shape = self.concat([self.period], tf.shape(value))
      last_values = tf.zeros(shape, value.dtype)

    new_last_values = shift(value, last_values)
    max_value = tf.cond(
      tf.less(iteration, self.period - 1),
      lambda: tf.reduce_max(new_last_values[:iteration + 1]),
      lambda: tf.reduce_max(new_last_values)
    )
    return max_value, (new_last_values, iteration + 1), (last_values, iteration)
