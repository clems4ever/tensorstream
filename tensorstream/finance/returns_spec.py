import numpy as np
import tensorflow as tf

from tensorstream.finance.returns import Return, LogarithmicReturn
from tensorstream.tests import TestCase

class ReturnSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(self.from_test_res('returns.ods', __file__))

  def test_return_single_dim(self):
    single_dim_ts = self.sheets['Sheet1']
    prices = tf.placeholder(tf.float32)
    return_ = Return(5)

    return_ts, _, _ = return_(prices)
    
    with tf.Session() as sess:
      output = sess.run(return_ts, {
        prices: single_dim_ts['Prices']
      })

    np.testing.assert_almost_equal(output,
      single_dim_ts['Return 5d'].values, decimal=4)


class LogarithmicReturnSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(self.from_test_res('logarithmic_returns.ods', __file__))

  def test_log_return_single_dim(self):
    single_dim_ts = self.sheets['Sheet1']
    prices = tf.placeholder(tf.float32)
    log_return = LogarithmicReturn(5)
    log_return_ts, _, _ = log_return(prices)
    
    with tf.Session() as sess:
      output = sess.run(log_return_ts, {
        prices: single_dim_ts['Prices']
      })

    np.testing.assert_almost_equal(output,
      single_dim_ts['Log Return 5d'].values, decimal=4)

