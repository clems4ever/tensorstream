import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.average_true_range import AverageTrueRange

class AverageTrueRangeSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('average_true_range.csv', __file__)).astype('float32')

  def test_average_true_range(self):
    atr = AverageTrueRange(14)
    close_prices = tf.placeholder(tf.float32)
    low_prices = tf.placeholder(tf.float32)
    high_prices = tf.placeholder(tf.float32)

    atr_ts, _ = atr(
      (
        close_prices,
        low_prices,
        high_prices
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
