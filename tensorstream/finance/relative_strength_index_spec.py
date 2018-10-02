import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.relative_strength_index import RelativeStrengthIndex

class RelativeStrengthIndexSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('relative_strength_index.ods', __file__))

  def test_rsi(self):
    sheet = self.sheets['relative_strength_index']
    rsi = RelativeStrengthIndex(14)
    values = tf.placeholder(tf.float32)
    rsi_ts, _, _ = rsi(values)

    with tf.Session() as sess:
      output = sess.run(rsi_ts, {
        values: sheet['Close'],
      })

    np.testing.assert_almost_equal(output,
      sheet['RSI'].values, decimal=3)
