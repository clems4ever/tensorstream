import numpy as np
import tensorflow as tf

from tensorstream.trading.supertrend import Supertrend

from tensorstream.tests import TestCase

class SupertrendSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('supertrend.csv', __file__)).astype('float32')

  def test_supertrend(self):
    supertrend = Supertrend(10, 3)
    close_prices = tf.placeholder(tf.float32)
    low_prices = tf.placeholder(tf.float32)
    high_prices = tf.placeholder(tf.float32)

    supertrend_ts, _ = supertrend(
      inputs=(close_prices, low_prices, high_prices)
    )

    with tf.Session() as sess:
      output = sess.run(supertrend_ts, {
        close_prices: self.input_ts['close'],
        low_prices: self.input_ts['low'],
        high_prices: self.input_ts['high'],
      })

    np.testing.assert_almost_equal(output,
      self.input_ts['Supertrend'].values, decimal=3)
