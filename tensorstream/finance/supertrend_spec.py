import numpy as np
import tensorflow as tf

from tensorstream.finance.supertrend import Supertrend

from tensorstream.tests import TestCase

class SupertrendSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('supertrend.ods', __file__))

  def test_supertrend(self):
    sheet = self.sheets['supertrend']
    supertrend = Supertrend(10, 3)
    close_prices = tf.placeholder(tf.float32)
    low_prices = tf.placeholder(tf.float32)
    high_prices = tf.placeholder(tf.float32)

    supertrend_ts, _, _ = supertrend(
      inputs=(close_prices, low_prices, high_prices)
    )

    with tf.Session() as sess:
      output = sess.run(supertrend_ts, {
        close_prices: sheet['close'],
        low_prices: sheet['low'],
        high_prices: sheet['high'],
      })

    np.testing.assert_almost_equal(output,
      sheet['Supertrend'].values, decimal=3)
