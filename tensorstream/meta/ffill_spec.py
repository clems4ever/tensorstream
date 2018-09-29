import numpy as np
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.meta.ffill import FFill
from tensorstream.tests import TestCase

class Identity(Streamable):
  def properties(self, x):
    return x, ()
  def step(self, x):
    return x, ()

class FfillWithoutStateSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('ffill.ods', __file__))

  def test_ffill_without_state(self):
    s = self.sheets['Sheet1'].astype('float32')

    values = s['Value 0']
    expected = s['Ffill 0']

    values_ph = tf.placeholder(tf.float32)
    op = FFill(Identity())

    ffill, _ = op(inputs=values_ph)

    with tf.Session() as sess:
      output = sess.run(ffill, {
        values_ph: values
      })

    np.testing.assert_almost_equal(output,
      expected.values, decimal=3)

class Lag(Streamable):
  def properties(self, x):
    return x, tf.constant(0.0)
  def step(self, x, last_value):
    return last_value, x

class FfillWithStateSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('ffill.ods', __file__))

  def test_ffill_with_state(self):
    s = self.sheets['Lag'].astype('float32')

    values = s['Value 0']
    expected = s['Ffill 0']

    values_ph = tf.placeholder(tf.float32)
    op = FFill(Lag())

    ffill, _ = op(values_ph)

    with tf.Session() as sess:
      output = sess.run(ffill, {
        values_ph: values
      })

    np.testing.assert_almost_equal(output,
      expected.values, decimal=3)

