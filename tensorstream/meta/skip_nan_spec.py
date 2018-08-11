import numpy as np
import tensorflow as tf

from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.meta.skip_nan import SkipNan
from tensorstream.meta.map import Map
from tensorstream.meta.factory import Factory
from tensorstream.trading.moving_average import SimpleMovingAverage
from tensorstream.trading.returns import Returns
from tensorstream.tests import TestCase

class SkipNanSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('skip_nan.ods', __file__))

  def test_simple_skip_nan(self):
    s = self.sheets['vectorize']

    values = s['Value 0'].replace(r'\s*', np.nan, regex=True)
    sma_outputs = s['SMA4 0'].replace(r'\s*', np.nan, regex=True)
    masks = s['Mask 0']

    values_ph = tf.placeholder(tf.float32)
    v = SkipNan(SimpleMovingAverage(4))
    o, _ = stream_to_tensor(v(Stream(values_ph)))

    with tf.Session() as sess:
      output = sess.run(o, {
        values_ph: values
      })

    np.testing.assert_almost_equal(output[0],
      sma_outputs.values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      masks.values, decimal=3)

  def test_map_skip_without_holes(self):
    s = self.sheets['vectorize']

    values = s[['Value 0', 'Value 1', 'Value 2']].replace(r'\s*', np.nan, regex=True)
    sma_outputs = s[['SMA4 0', 'SMA4 1', 'SMA4 2']].replace(r'\s*', np.nan, regex=True)
    masks = s[['Mask 0', 'Mask 1', 'Mask 2']]

    values_ph = tf.placeholder(tf.float32)
    v = Map(SkipNan(SimpleMovingAverage(4)), 3)
    o, _ = stream_to_tensor(v(Stream(values_ph)))

    with tf.Session() as sess:
      output = sess.run(o, {
        values_ph: values
      })

    np.testing.assert_almost_equal(output[0],
      sma_outputs.values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      masks.values, decimal=3)

  def test_map_skip_with_holes(self):
    s = self.sheets['holes']

    values = s[['Value 0', 'Value 1', 'Value 2']].replace(r'\s*', np.nan, regex=True)
    sma_outputs = s[['SMA4 0', 'SMA4 1', 'SMA4 2']].replace(r'\s*', np.nan, regex=True)
    masks = s[['Mask 0', 'Mask 1', 'Mask 2']]

    values_ph = tf.placeholder(tf.float32, shape=[None, 3])
    sma4 = SimpleMovingAverage(4)
    v = Map(SkipNan(sma4), 3)
    o, _ = stream_to_tensor(v(Stream(values_ph)))

    with tf.Session() as sess:
      output = sess.run(o, {
        values_ph: values
      })

    np.testing.assert_almost_equal(output[0],
      sma_outputs.values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      masks.values, decimal=3)
