import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SimpleMovingAverage(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = (
      tf.constant(0),
      tf.fill((period,) + self.shape, math.nan)
    )

  # Only one value expected, the price
  def step(self, value, iteration, buffer_):
    new_buffer = roll(value, buffer_)
    new_value = tf.cond(iteration < self.period - 1, 
      lambda: tf.fill(self.shape, math.nan),
      lambda: tf.reduce_sum(new_buffer, axis=0) / self.period)
    return new_value, (iteration + 1, new_buffer)

class ExponentialMovingAverage(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.k = 2.0 / (period + 1.0)
    self.sma = SimpleMovingAverage(period, dtype=dtype, shape=shape)
    self.initial_state = (
      tf.fill(shape, math.nan),
      self.sma.initial_state
    )

  # One value expected, the price
  def step(self, value, last_ema, last_sma_state):
    def warmup():
      return self.sma(value, state=last_sma_state)
    def nominal():
      new_ema = value * self.k + last_ema * (1.0 - self.k)
      return new_ema, last_sma_state

    new_ema, new_sma_state = tf.cond(
      tf.reduce_any(tf.is_nan(last_ema)),
      warmup,
      nominal
    )
    return new_ema, (new_ema, new_sma_state)

class RollingMovingAverage(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.sma = SimpleMovingAverage(period, dtype, shape)
    self.initial_state = (
      tf.fill(shape, math.nan),
      self.sma.initial_state
    )

  def step(self, value, last_rma, last_sma_state):
    def warmup():
      return self.sma(value, state=last_sma_state)

    def nominal():
      new_rma = (last_rma * (self.period - 1) + value) / self.period
      return new_rma, last_sma_state

    new_rma, new_sma_state = tf.cond(
      tf.reduce_any(tf.is_nan(last_rma)), warmup, nominal)

    return new_rma, (new_rma, new_sma_state)
