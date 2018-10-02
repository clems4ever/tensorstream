import numpy as np
import tensorflow as tf

from tensorstream.finance.bollinger_bands import BollingerBands
from tensorstream.tests import TestCase

class BollingerBandsSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('bollinger_bands.ods', __file__))

  def test_bollinger_bands(self):
    sheet = self.sheets['bollinger_bands']

    values = tf.placeholder(tf.float32)
    bollinger_bands = BollingerBands(20, 2)
    bb_ts, _, _ = bollinger_bands(values)

    with tf.Session() as sess:
      output = sess.run(bb_ts, {
        values: sheet['Price'],
      })

    np.testing.assert_almost_equal(output[0],
      sheet['Lower Band'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      sheet['Middle Band'].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      sheet['Upper Band'].values, decimal=3)
