import numpy as np
import tensorflow as tf
from tensorstream.finance.momentum import Momentum
from tensorstream.tests import TestCase

class MomentumSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_ods(
      self.from_test_res('momentum_5.ods', __file__))["Sheet1"]
    self.input_ts.replace(r'\s*', np.nan, regex=True)

  def test_lag(self):
    values = tf.placeholder(tf.float32)
    buffer_ts, _ = Momentum(5)(values)

    with tf.Session() as sess:
      output = sess.run(buffer_ts, { values: self.input_ts['Close'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['Momentum'].values, decimal=3)

