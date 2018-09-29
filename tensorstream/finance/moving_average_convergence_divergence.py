import tensorflow as tf

from tensorstream.finance.moving_average import ExponentialMovingAverage
from tensorstream.placeholder import Placeholder
from tensorstream.streamable import Streamable

class MovingAverageConvergenceDivergence(Streamable):
  def __init__(self, slow, fast, macd):
    super().__init__()
    self.macd_period = macd
    self.slow_period = slow
    self.fast_period = fast
    self.ema_slow = ExponentialMovingAverage(slow)
    self.ema_fast = ExponentialMovingAverage(fast)
    self.ema_macd = ExponentialMovingAverage(macd)

  def properties(self, value):
    ema_slow_ph, ema_slow_init_state = self.ema_slow.properties(value)
    ema_fast_ph, ema_fast_init_state = self.ema_fast.properties(value)
    ema_macd_ph, ema_macd_init_state = self.ema_macd.properties(value)

    ph = Placeholder(value.dtype, value.shape)
    return (ph,) * 5, (
      tf.constant(0),
      ema_slow_init_state,
      ema_fast_init_state,
      ema_macd_init_state
    )

  def _diff(self, value1, value2, iteration,
    starting_iteration):
    return tf.cond(
      iteration < starting_iteration,
      lambda: tf.zeros(tf.shape(value1), dtype=value1.dtype),
      lambda: value1 - value2)

  def step(self, value, iteration, ema_slow_state,
    ema_fast_state, ema_macd_state):

    ema_slow, new_ema_slow_state = self.ema_slow(
      value, state=ema_slow_state, streamable=False)
    ema_fast, new_ema_fast_state = self.ema_fast(
      value, state=ema_fast_state, streamable=False)

    macd = self._diff(
      ema_fast, ema_slow, iteration, self.slow_period - 1)

    signal, new_ema_macd_state = tf.cond(
      iteration < self.slow_period - 1,
      lambda: (tf.zeros(tf.shape(value), dtype=value.dtype), ema_macd_state),
      lambda: self.ema_macd(macd, state=ema_macd_state, streamable=False)
    )

    histogram = self._diff(macd, signal, iteration, 
      self.slow_period + self.macd_period - 2)

    return (
      ema_slow,
      ema_fast,
      macd,
      signal,
      histogram
    ), (
      iteration + 1,
      new_ema_slow_state,
      new_ema_fast_state,
      new_ema_macd_state
    )


