import tensorflow as tf

from tensorstream.common import shift
from tensorstream.streamable import Streamable

class Extremum(Streamable):
  def __init__(self):
    super().__init__()

  def initial_state(sef, value):
    return (
      tf.zeros(tf.shape(value), dtype=value.dtype),
      tf.constant(True)
    )

class GlobalMinimum(Extremum):
  def __init__(self):
    super().__init__()

  def step(self, value, global_min, is_first_iteration):
    new_min = tf.cond(
      tf.logical_or(
        is_first_iteration,
        tf.less(value, global_min)
      ),
      lambda: value, lambda: global_min)

    # TODO: remove tf.constant around bool. streamable needs
    # a Tensor instead of raw value for now but it can be
    # automatically converted.
    return new_min, (new_min, tf.constant(False))

class GlobalMaximum(Extremum):
  def __init__(self):
    super().__init__()

  def step(self, value, global_max, is_first_iteration):
    new_max = tf.cond(
      tf.logical_or(
        is_first_iteration,
        tf.greater(value, global_max)
      ),
      lambda: value, lambda: global_max)
    # TODO: remove tf.constant around bool. streamable needs
    # a Tensor instead of raw value for now but it can be
    # automatically converted.
    return new_max, (new_max, tf.constant(False))

class LocalExtremum(Extremum):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def initial_state(self, value):
    shape = self.concat([self.period], tf.shape(value))
    return (
      tf.zeros(shape, dtype=value.dtype),
      tf.constant(0)
    )

class LocalMinimum(LocalExtremum):
  def __init__(self, period):
    super().__init__(period)

  def step(self, value, last_values, iteration):
    new_last_values = shift(value, last_values)
    min_value = tf.cond(
      tf.less(iteration, self.period - 1),
      lambda: tf.reduce_min(new_last_values[:iteration + 1]),
      lambda: tf.reduce_min(new_last_values)
    )
    return min_value, (new_last_values, iteration + 1)

class LocalMaximum(LocalExtremum):
  def __init__(self, period):
    super().__init__(period)

  def step(self, value, last_values, iteration):
    new_last_values = shift(value, last_values)
    max_value = tf.cond(
      tf.less(iteration, self.period - 1),
      lambda: tf.reduce_max(new_last_values[:iteration + 1]),
      lambda: tf.reduce_max(new_last_values)
    )
    return max_value, (new_last_values, iteration + 1)
