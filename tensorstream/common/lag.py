import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Lag(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def initial_state(self, value):
    return tf.zeros(
      self.concat([self.period], tf.shape(value))
    )

  def step(self, value, buffer_state):
    new_value = buffer_state[-1]
    new_buffer_state = roll(value, buffer_state)
    return new_value, new_buffer_state
