import tensorflow as tf

from tensorstream.streamable import Streamable

class HeikinAshi(Streamable):
  """
  Heikin Ashi implementation.
  https://www.investopedia.com/trading/heikin-ashi-better-candlestick/
  http://viewpdxblue.com/2016/02/29/heiken-ashi-emini-trend-direction-excel/

  """
  def step(self, open_price, high_price, low_price, close_price,
          prev_open_price=None, prev_close_price=None):
    """
    At the first step, prev_open_price = prev_close_prise = -1 and output = input     
    """
    if prev_open_price is None:
      prev_open_price = tf.zeros(tf.shape(open_price), open_price.dtype)
    if prev_close_price is None:
      prev_close_price = tf.zeros(tf.shape(close_price), close_price.dtype)

    ha_close_price = tf.cond(tf.equal(prev_open_price, 0),
                            lambda : close_price,
                            lambda : tf.reduce_sum([close_price, 
                                                    low_price,
                                                    high_price,
                                                    open_price]) / 4,
                            )

    ha_open_price =  tf.cond(tf.equal(prev_open_price, 0),
                            lambda: open_price,
                            lambda: (prev_open_price + prev_close_price) / 2)

    ha_low_price = tf.cond(tf.equal(prev_open_price, 0),
                          lambda : low_price,
                          lambda : tf.reduce_min(
                            tf.stack([low_price, ha_close_price, ha_open_price]))
                          )

    ha_high_price = tf.cond(tf.equal(prev_open_price, 0),
                            lambda : high_price,
                            lambda : tf.reduce_max(
                              tf.stack([high_price, ha_close_price, ha_open_price]))
                            )

    new_value = (ha_open_price, ha_high_price, ha_low_price, ha_close_price)
    return new_value, (open_price, close_price), (prev_open_price, prev_close_price)
  
