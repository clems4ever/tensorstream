import math
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.finance.moving_average import RollingMovingAverage

class AverageTrueRange(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.rma = RollingMovingAverage(period, dtype=dtype, shape=shape)
    self.initial_state = (tf.fill(shape, math.nan), self.rma.initial_state)

  def step(self, close_price, low_price, high_price, last_close, last_rma_state):
    hl = high_price - low_price
    hcp = tf.abs(high_price - last_close)
    lcp = tf.abs(low_price - last_close)
    tr = tf.reduce_max(tf.stack([hl, hcp, lcp]))

    atr, new_rma_state = self.rma(tr, state=last_rma_state, streamable=False)
    return atr, (close_price, new_rma_state)
