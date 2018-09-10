import tensorflow as tf
import numpy as np

from tensorstream.tests import TestCase
from tensorstream.finance.sharpe_ratio import SharpeRatio

class SharpeRatioSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('sharpe_ratio.ods', __file__))

  def run_single_dim(self, sheet_name):
    s = self.sheets[sheet_name]
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

