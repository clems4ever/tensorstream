import math
import tensorflow as tf

from tensorstream.streamable import Streamable

class HeikinAshi(Streamable):
  """
  Heikin Ashi implementation.
  https://www.investopedia.com/trading/heikin-ashi-better-candlestick/
  http://viewpdxblue.com/2016/02/29/heiken-ashi-emini-trend-direction-excel/

  """

  def __init__(self, dtype=tf.float32, shape=()):
    super().__init__((dtype,) * 4, (shape,)*4)

    # previous open and close are initialized to NaN for the first step
    self.initial_state = (tf.constant(math.nan,dtype), tf.constant(math.nan, dtype))

  def step(self, open_price, high_price, low_price, close_price,
          prev_open_price, prev_close_price):
    """
    At the first step, prev_open_price = prev_close_prise = -1 and output = input     
    """

    ha_close_price = tf.cond(tf.is_nan(prev_open_price),
                            lambda : close_price,
                            lambda : tf.reduce_sum([close_price, 
                                                    low_price,
                                                    high_price,
                                                    open_price]) / 4,
                            )

    ha_open_price =  tf.cond(tf.is_nan(prev_open_price),
                            lambda: open_price,
                            lambda: (prev_open_price + prev_close_price) / 2)

    ha_low_price = tf.cond(tf.is_nan(prev_open_price),
                          lambda : low_price,
                          lambda : tf.reduce_min(
                            tf.stack([low_price, close_price, open_price]))
                          )

    ha_high_price = tf.cond(tf.is_nan(prev_open_price),
                            lambda : high_price,
                            lambda : tf.reduce_max(
                              tf.stack([high_price, close_price, open_price]))
                            )

    new_value = (ha_open_price, ha_high_price, ha_low_price, ha_close_price)
    return new_value, (open_price, close_price)
  
