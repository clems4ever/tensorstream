import math
import tensorflow as tf

from tensorstream.operators import Streamable, roll

class Lag(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = tf.fill((period,) + shape, math.nan)

  def step(self, value, buffer_state):
    new_value = buffer_state[-1]
    new_buffer_state = roll(value, buffer_state)
    return new_value, new_buffer_state
