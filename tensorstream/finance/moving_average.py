import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class SimpleMovingAverage(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def step(self, value, iteration=None, prev_buffer=None):
    if iteration is None:
      iteration = tf.constant(0)
    if prev_buffer is None:
      shape = self.concat([self.period], tf.shape(value))
      prev_buffer = tf.zeros(shape, value.dtype)

    new_buffer = roll(value, prev_buffer)
    new_value = tf.cond(iteration < self.period - 1, 
      lambda: tf.zeros(tf.shape(value), dtype=value.dtype),
      lambda: tf.reduce_sum(new_buffer, axis=0) / self.period)
    return new_value, (iteration + 1, new_buffer), (iteration, prev_buffer)

class ExponentialMovingAverage(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period
    self.k = 2.0 / (period + 1.0)
    self.sma = SimpleMovingAverage(period)

  # One value expected, the price
  def step(self, value, iteration=None, last_ema=None,
    last_sma_state=None):

    if iteration is None:
      iteration = tf.constant(0)
    if last_ema is None:
      last_ema = tf.zeros(tf.shape(value), value.dtype)

    sma, next_sma_state, sma_init_state = self.sma(
      value, state=last_sma_state, streamable=False)

    if last_sma_state is None:
      last_sma_state = sma_init_state

    new_ema = value * self.k + last_ema * (1.0 - self.k)

    new_ema, new_sma_state = tf.cond(
      iteration < self.period,
      lambda: (sma, next_sma_state),
      lambda: (new_ema, last_sma_state)
    )
    return new_ema, (
      iteration + 1, new_ema, new_sma_state
    ), (
      iteration, last_ema, last_sma_state
    )

class RollingMovingAverage(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period
    self.sma = SimpleMovingAverage(period)

  def step(self, value, iteration=None, last_rma=None, last_sma_state=None):
    if iteration is None:
      iteration = tf.constant(0)
    if last_rma is None:
      last_rma = tf.zeros(tf.shape(value), value.dtype)

    sma, next_sma_state, init_sma_state =  self.sma(
      value, state=last_sma_state, streamable=False)

    if last_sma_state is None:
      last_sma_state = init_sma_state

    new_rma = (last_rma * (self.period - 1) + value) / self.period

    new_rma, new_sma_state = tf.cond(
      iteration < self.period,
      lambda: (sma, next_sma_state),
      lambda: (new_rma, last_sma_state)
    )
    return new_rma, (
      iteration + 1, new_rma, new_sma_state
    ), (
      iteration, last_rma, last_sma_state
    )
