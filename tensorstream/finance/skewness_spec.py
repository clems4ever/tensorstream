import numpy as np
import tensorflow as tf

from tensorstream.finance.skewness import Skewness
from tensorstream.tests import TestCase

class SkewnessSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('skewness.ods', __file__))

  def test_single_dim(self):
    s = self.sheets['Sheet1']
    s10 = Skewness(10)
    prices = tf.placeholder(tf.float32)
    s10_ts, _, _ = s10(prices)
    
    with tf.Session() as sess:
      output = sess.run(s10_ts, {
        prices: s['Return'],
      })

    np.testing.assert_almost_equal(output,
      s['Skewness 10D'].values, decimal=3)

