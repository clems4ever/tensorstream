import tensorflow as tf
import numpy as np

from tensorstream.tests import TestCase
from tensorstream.trading.sharpe_ratio import SharpeRatio

class SharpeRatioSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('sharpe_ratio.ods', __file__))

  def run_single_dim(self, sheet_name):
    s = self.sheets[sheet_name].replace(r'\s*', np.nan, regex=True)
    sr10 = SharpeRatio(10)
    prices = tf.placeholder(tf.float32)
    risk_free_rates = tf.placeholder(tf.float32)
    sr10_ts, _ = sr10(inputs=(prices, risk_free_rates))
    
    with tf.Session() as sess:
      output = sess.run(sr10_ts, {
        prices: s['Return'],
        risk_free_rates: s['Risk free rate']
      })

    np.testing.assert_almost_equal(output,
      s['SR10'].values, decimal=3)

  def test_single_dim_zero_rfr(self):
    self.run_single_dim('single_dim_zero_rfr')

  def test_single_dim_small_rfr(self):
    self.run_single_dim('single_dim_small_rfr')

  def test_multi_dim(self):
    s = self.sheets['multi_dim'].replace(r'\s*', np.nan, regex=True)
    sr10 = SharpeRatio(10, dtype=tf.float32, shape=(2,))
    prices = tf.placeholder(tf.float32, shape=[None, 2])
    risk_free_rates = tf.placeholder(tf.float32, shape=[None, 2])
    sr10_ts, _ = sr10(inputs=(prices, risk_free_rates))

    prices_ts = s[['Return 0', 'Return 1']]
    rfr_ts = s[['Risk free rate 0', 'Risk free rate 1']]
    expected_outputs_ts = s[['SR10 0', 'SR10 1']]
    
    with tf.Session() as sess:
      output = sess.run(sr10_ts, {
        prices: prices_ts,
        risk_free_rates: rfr_ts
      })

    np.testing.assert_almost_equal(output,
      expected_outputs_ts.values, decimal=3)

