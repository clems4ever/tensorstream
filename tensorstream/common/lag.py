import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Lag(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def step(self, value, buffer_state=None):
    if buffer_state is None:
      shape = self.concat([self.period], tf.shape(value))
      buffer_state = tf.zeros(shape, value.dtype)

    new_value = buffer_state[-1]
    new_buffer_state = roll(value, buffer_state)
    return new_value, new_buffer_state, buffer_state
