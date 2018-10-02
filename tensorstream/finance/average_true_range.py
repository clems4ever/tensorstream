import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.finance.moving_average import RollingMovingAverage

class AverageTrueRange(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period
    self.rma = RollingMovingAverage(period)

  def step(self, close_price, low_price, high_price,
    last_close=None, last_rma_state=None):

    if last_close is None:
      last_close = tf.zeros(tf.shape(close_price), close_price.dtype)

    hl = high_price - low_price
    hcp = tf.where(
      tf.equal(last_close, 0),
      tf.zeros(tf.shape(last_close), dtype=last_close.dtype),
      tf.abs(high_price - last_close)
    )
    lcp = tf.where(
      tf.equal(last_close, 0),
      tf.zeros(tf.shape(last_close), dtype=last_close.dtype),
      tf.abs(low_price - last_close)
    )
    tr = tf.reduce_max(tf.stack([hl, hcp, lcp]))

    atr, new_rma_state, rma_init_state = self.rma(tr, state=last_rma_state, streamable=False)
    return atr, (close_price, new_rma_state), (last_close, rma_init_state)
