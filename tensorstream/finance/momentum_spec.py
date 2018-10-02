import numpy as np
import tensorflow as tf
from tensorstream.finance.momentum import Momentum
from tensorstream.tests import TestCase

class MomentumSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('momentum_5.ods', __file__))

  def test_momentum_single_dim(self):
    values = tf.placeholder(tf.float32)
    momentum_ts, _, _ = Momentum(5)(values)

    input_ts = self.sheets['Sheet1']

    with tf.Session() as sess:
      output = sess.run(momentum_ts, { values: input_ts['Close'] })

    np.testing.assert_almost_equal(output,
      input_ts['Momentum'].values, decimal=3)

