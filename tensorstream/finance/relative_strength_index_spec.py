import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.relative_strength_index import RelativeStrengthIndex

class RelativeStrengthIndexSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('relative_strength_index.csv', __file__))\
        .astype('float32')

  def test_rsi(self):
    rsi = RelativeStrengthIndex(14)
    values = tf.placeholder(tf.float32)
    rsi_ts, _ = rsi(values)

    with tf.Session() as sess:
      output = sess.run(rsi_ts, {
        values: self.input_ts['Close'],
      })

    np.testing.assert_almost_equal(output,
      self.input_ts['RSI'].values, decimal=3)
