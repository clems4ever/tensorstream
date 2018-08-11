import numpy as np
import tensorflow as tf

from tensorstream.common.lag import Lag
from tensorstream.tests import TestCase

class LagSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('lag.csv', __file__)).astype('float32')

  def test_lag(self):
    values = tf.placeholder(tf.float32)
    buffer_ts, _ = Lag(3)(values)

    with tf.Session() as sess:
      output = sess.run(buffer_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['Delayed'].values, decimal=3)
