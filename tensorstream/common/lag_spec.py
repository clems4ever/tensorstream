import numpy as np
import tensorflow as tf

from tensorstream.common.lag import Lag
from tensorstream.tests import TestCase

class LagSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('lag.ods', __file__))

  def test_lag_single_dim(self):
    values = tf.placeholder(tf.float32)
    buffer_ts, _ = Lag(3)(values)

    input_ts = self.sheets['single_dim']

    with tf.Session() as sess:
      output = sess.run(buffer_ts, { values: input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      input_ts['Delayed'].values, decimal=3)

