import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.average_true_range import AverageTrueRange

class AverageTrueRangeSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('average_true_range.ods', __file__))

  def test_average_true_range(self):
    sheet = self.sheets['average_true_range']
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
        close_prices: sheet['Close'],
        low_prices: sheet['Low'],
        high_prices: sheet['High'],
      })

    np.testing.assert_almost_equal(output,
      sheet['ATR'].values, decimal=3)
