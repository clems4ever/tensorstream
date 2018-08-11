import numpy as np
import tensorflow as tf

from tensorstream.trading.returns import Returns
from tensorstream.tests import TestCase

class VariationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(self.from_test_res('returns.ods', __file__))

  def test_single_dim(self):
    single_dim_ts = self.sheets['single_dim']
    prices = tf.placeholder(tf.float32)
    variation1 = Returns(1)
    variation5 = Returns(5)

    variation1_ts, _ = variation1(prices)
    variation5_ts, _ = variation5(prices)
    
    with tf.Session() as sess:
      output = sess.run([variation1_ts, variation5_ts], {
        prices: single_dim_ts['Prices']
      })

    np.testing.assert_almost_equal(output[0],
      single_dim_ts['Variation 1d'].values, decimal=4)
    np.testing.assert_almost_equal(output[1],
      single_dim_ts['Variation 5d'].values, decimal=4)

  def test_multi_dim(self):
    single_dim_ts = self.sheets['multi_dim']
    prices_ts = single_dim_ts[['Prices 0', 'Prices 1']]
    var_ts = single_dim_ts[['Variation 1d 0', 'Variation 1d 1']]

    prices = tf.placeholder(tf.float32, shape=[None, 2])
    variation1 = Returns(1, shape=(2,))
    variation1_ts, _ = variation1(prices)
    
    with tf.Session() as sess:
      output = sess.run(variation1_ts, { prices: prices_ts })

    np.testing.assert_almost_equal(output, var_ts.values, decimal=4)

