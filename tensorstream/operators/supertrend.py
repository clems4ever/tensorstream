import math
import tensorflow as tf

from tensorstream.operators.average_true_range import AverageTrueRange
from tensorstream.operators import Streamable

class Supertrend(Streamable):
  def __init__(self, atr_period, factor, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.factor = factor
    self.atr = AverageTrueRange(atr_period)
    self.initial_state = (
      self.atr.initial_state,
      math.nan,
      math.nan,
      math.nan,
      tf.constant(0)
    )

  def step(self, close_price, low_price, high_price, 
           atr_state, last_close, last_trend_down, last_trend_up, last_trend):
    def default(atr):
      hl2 = (high_price + low_price) / 2.0
      up = hl2 - self.factor * atr
      down = hl2 + self.factor * atr

      trend_up = tf.cond(last_close > last_trend_up,
        lambda: tf.maximum(up, last_trend_up),
        lambda: up
      )

      trend_down = tf.cond(last_close < last_trend_down,
        lambda: tf.minimum(down, last_trend_down),
        lambda: down
      )

      trend = tf.case({
        close_price > last_trend_down: lambda: 1,
        close_price < last_trend_up: lambda: -1
      }, exclusive=True, default=lambda: tf.cond(tf.equal(last_trend, 0),
        lambda: 1,
        lambda: last_trend
      ))

      supertrend = tf.cond(tf.equal(trend, 1),
        lambda: trend_up,
        lambda: trend_down
      )

      return trend_down, trend_up, trend, supertrend
      
    def warmup():
      return math.nan, math.nan, 0, math.nan
    
    atr, atr_state = self.atr(close_price, low_price, high_price, state=atr_state)
    trend_down, trend_up, trend, supertrend = tf.cond(
      tf.is_nan(atr),
      lambda: warmup(),
      lambda: default(atr)
    )

    return supertrend, (
      atr_state,
      close_price,
      trend_down,
      trend_up,
      trend
    )
