import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SimpleMovingAverage(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def properties(self, value):
    shape = self.concat([self.period], value.shape)
    return value, (tf.constant(0), tf.zeros(dtype=value.dtype, shape=shape))

  def step(self, value, iteration, buffer_):
    new_buffer = roll(value, buffer_)
    new_value = tf.cond(iteration < self.period - 1, 
      lambda: tf.zeros(tf.shape(value), dtype=value.dtype),
      lambda: tf.reduce_sum(new_buffer, axis=0) / self.period)
    return new_value, (iteration + 1, new_buffer)

class ExponentialMovingAverage(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period
    self.k = 2.0 / (period + 1.0)
    self.sma = SimpleMovingAverage(period)

  def properties(self, value):
    ph, init_state = self.sma.properties(value)
    return value, (
      tf.zeros(value.shape, dtype=value.dtype),
      init_state,
      tf.constant(0)
    )

  # One value expected, the price
  def step(self, value, last_ema, last_sma_state, iteration):
    def warmup():
      return self.sma(value, state=last_sma_state, streamable=False)
    def nominal():
      new_ema = value * self.k + last_ema * (1.0 - self.k)
      return new_ema, last_sma_state

    new_ema, new_sma_state = tf.cond(
      iteration < self.period,
      warmup,
      nominal
    )
    return new_ema, (new_ema, new_sma_state, iteration + 1)

class RollingMovingAverage(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period
    self.sma = SimpleMovingAverage(period)

  def properties(self, value):
    ph, init_state = self.sma.properties(value)
    return value, (
      tf.zeros(value.dtype, dtype=value.dtype),
      init_state,
      tf.constant(0)
    )

  def step(self, value, last_rma, last_sma_state, iteration):
    def warmup():
      return self.sma(value, state=last_sma_state, streamable=False)

    def nominal():
      new_rma = (last_rma * (self.period - 1) + value) / self.period
      return new_rma, last_sma_state

    new_rma, new_sma_state = tf.cond(
      iteration < self.period,
      warmup,
      nominal
    )
    return new_rma, (new_rma, new_sma_state, iteration + 1)
