import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.finance.moving_average_convergence_divergence import MovingAverageConvergenceDivergence as MACD
from tensorstream.tests import TestCase

class MACDSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_average_convergence_divergence.ods', __file__))
  
  def test_single_dim(self):
    s = self.sheets['Sheet1']
    prices_ts = s['Close']

    expected_ema_slow = s['26 Day EMA']
    expected_ema_fast = s['12 Day EMA']
    expected_macd = s['MACD']
    expected_signal = s['Signal']
    expected_histogram = s['Histogram']
 
    macd_26_12_9 = MACD(26, 12, 9)
    prices = tf.placeholder(tf.float32)
    macd_ts, _, _ = macd_26_12_9(prices)

    with tf.Session() as sess:
      output_ts = sess.run(macd_ts, {
        prices: prices_ts
      })

    np.testing.assert_almost_equal(output_ts[0], 
      expected_ema_slow.values, decimal=3)
    np.testing.assert_almost_equal(output_ts[1], 
      expected_ema_fast.values, decimal=3)
    np.testing.assert_almost_equal(output_ts[2], 
      expected_macd.values, decimal=3)
    np.testing.assert_almost_equal(output_ts[3], 
      expected_signal.values, decimal=3)
    np.testing.assert_almost_equal(output_ts[4], 
      expected_histogram.values, decimal=3)
    
