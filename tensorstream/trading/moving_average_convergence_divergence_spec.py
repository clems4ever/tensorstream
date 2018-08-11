import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.trading.moving_average_convergence_divergence import MovingAverageConvergenceDivergence as MACD
from tensorstream.tests import TestCase

class MACDSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_average_convergence_divergence.ods', __file__))
  
  def test_single_dim(self):
    s = self.sheets['single_dim']
    prices_ts = s['Close']

    expected_ema_slow = s['26 Day EMA']
    expected_ema_fast = s['12 Day EMA']
    expected_macd = s['MACD']
    expected_signal = s['Signal']
    expected_histogram = s['Histogram']
 
    macd_26_12_9 = MACD(26, 12, 9)
    prices = tf.placeholder(tf.float32)
    macd_ts, _ = stream_to_tensor(macd_26_12_9(Stream(prices)))

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
    
  def test_multi_dim(self):
    s = self.sheets['multi_dim'].replace(r'\s*', np.nan, regex=True)
    prices_ts = s[['Close 0', 'Close 1']]

    expected_ema_slow = s[['26 Day EMA 0', '26 Day EMA 1']]
    expected_ema_fast = s[['12 Day EMA 0', '12 Day EMA 1']]
    expected_macd = s[['MACD 0', 'MACD 1']]
    expected_signal = s[['Signal 0', 'Signal 1']]
    expected_histogram = s[['Histogram 0', 'Histogram 1']]

    prices = tf.placeholder(tf.float32, shape=[None, 2])
    macd_26_12_9 = MACD(26, 12, 9, dtype=tf.float32, shape=(2,))
    macd_ts, _ = stream_to_tensor(macd_26_12_9(Stream(prices)))

    with tf.Session() as sess:
      output_ts = sess.run(macd_ts, { prices: prices_ts })

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
    
