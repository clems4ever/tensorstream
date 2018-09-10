import tensorflow as tf

from tensorstream.helpers.map_fn import map_fn
from tensorstream.streamable import MetaStreamable

class SetDuring(MetaStreamable):
  """
  Set a specific value during a certain period
  """
  def __init__(self, value, period):
    super().__init__(0)
    self.value = value
    self.period = period

  def step(self, inputs, iteration):
    inp = map_fn(inputs, [inputs], lambda x: tf.cond(
      tf.less(iteration, self.period),
      lambda: tf.fill(tf.shape(x), self.value),
      lambda: x
    ))
    return inp, iteration + 1
