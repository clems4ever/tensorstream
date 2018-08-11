import numpy as np
import tensorflow as tf

from tensorstream.common.extremum import GlobalMinimum, GlobalMaximum, LocalMinimum, LocalMaximum
from tensorstream.tests import TestCase

class ExtremumSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('extremum.csv', __file__))\
        .fillna(0.0).astype('float32')

  def test_global_min(self):
    global_minimum = GlobalMinimum()
    values = tf.placeholder(tf.float32)
    global_minimum_ts, _ = global_minimum(values)

    with tf.Session() as sess:
      output = sess.run(global_minimum_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['GlobalMin'].values, decimal=3)

  def test_global_max(self):
    global_maximum = GlobalMaximum()
    values = tf.placeholder(tf.float32)
    global_maximum_ts, _ = global_maximum(values)

    with tf.Session() as sess:
      output = sess.run(global_maximum_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['GlobalMax'].values, decimal=3)

  def test_local_min(self):
    local_minimum = LocalMinimum(5)
    values = tf.placeholder(tf.float32)
    local_minimum_ts, _ = local_minimum(values)

    with tf.Session() as sess:
      output = sess.run(local_minimum_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['LocalMin'].values, decimal=3)

  def test_local_max(self):
    local_maximum = LocalMaximum(5)
    values = tf.placeholder(tf.float32)
    local_maximum_ts, _ = local_maximum(values)

    with tf.Session() as sess:
      output = sess.run(local_maximum_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['LocalMax'].values, decimal=3)
