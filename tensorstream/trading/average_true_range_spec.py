import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.trading.average_true_range import AverageTrueRange

class AverageTrueRangeSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('average_true_range.csv', __file__)).astype('float32')

  def test_average_true_range(self):
    atr = AverageTrueRange(14)
    close_prices = tf.placeholder(tf.float32)
    low_prices = tf.placeholder(tf.float32)
    high_prices = tf.placeholder(tf.float32)

    close_prices_stream = Stream(close_prices)
    low_prices_stream = Stream(low_prices)
    high_prices_stream = Stream(high_prices)

    atr_ts, _ = stream_to_tensor(
      atr(
        close_prices_stream,
        low_prices_stream,
        high_prices_stream
      )
    )
    
    with tf.Session() as sess:
      output = sess.run(atr_ts, {
        close_prices: self.input_ts['Close'],
        low_prices: self.input_ts['Low'],
        high_prices: self.input_ts['High'],
      })

    np.testing.assert_almost_equal(output,
      self.input_ts['ATR'].values, decimal=3)
