import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Returns(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = (0, tf.fill((period,) + shape, math.nan))

  def step(self, value, iteration, old_buffer):
    old_value = old_buffer[self.period - 1]
    new_buffer = roll(value, old_buffer)

    variation = tf.cond(
      iteration < self.period - 1,
      lambda: tf.fill(self.shape, math.nan),
      lambda: value / old_value - tf.constant(1.0)
    )
    return variation, (iteration + 1, new_buffer)
