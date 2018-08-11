import math
import tensorflow as tf

from tensorstream.trading.moving_average import ExponentialMovingAverage
from tensorstream.streamable import Streamable

class MovingAverageConvergenceDivergence(Streamable):
  def __init__(self, slow, fast, macd, dtype=tf.float32, shape=()):
    super().__init__((dtype,) * 5, (shape,) * 5)
    self.macd_period = macd
    self.slow_period = slow
    self.fast_period = fast
    self._ema_slow = ExponentialMovingAverage(self.slow_period, dtype=dtype, shape=shape)
    self._ema_fast = ExponentialMovingAverage(self.fast_period, dtype=dtype, shape=shape)
    self._ema_macd = ExponentialMovingAverage(self.macd_period, dtype=dtype, shape=shape)

    self.initial_state = (
      tf.constant(0),
      self._ema_slow.initial_state,
      self._ema_fast.initial_state,
      self._ema_macd.initial_state
    )

  def _diff(self, value1, value2, iteration, starting_iteration):
    return tf.cond(iteration < starting_iteration,
                   lambda: tf.fill(self.shape[0], math.nan),
                   lambda: value1 - value2)

  def step(self, value, iteration, ema_slow_state, ema_fast_state, ema_macd_state):
    ema_slow, new_ema_slow_state = self._ema_slow(value, state=ema_slow_state)
    ema_fast, new_ema_fast_state = self._ema_fast(value, state=ema_fast_state)

    macd = self._diff(ema_fast, ema_slow, iteration, self.slow_period - 1)

    signal, new_ema_macd_state = tf.cond(iteration < self.slow_period - 1,
      lambda: (tf.fill(self.shape[0], math.nan), ema_macd_state),
      lambda: self._ema_macd(macd, state=ema_macd_state))

    histogram = self._diff(macd, signal, iteration, 
      self.slow_period + self.macd_period - 2)

    return (ema_slow, ema_fast, macd, signal, histogram), (
      iteration + 1,
      new_ema_slow_state,
      new_ema_fast_state,
      new_ema_macd_state
    )


