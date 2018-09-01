import numpy as np
import tensorflow as tf

from tensorstream.meta.compose import Compose
from tensorstream.meta.factory import Factory
from tensorstream.meta.map import Map
from tensorstream.finance.moving_average import SimpleMovingAverage
from tensorstream.tests import TestCase

class MapSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('map.ods', __file__))

  def test_map_skip_without_holes(self):
    s = self.sheets['Sheet1']

    values = s[['Value 0', 'Value 1', 'Value 2']].replace(r'\s*', np.nan, regex=True)
    sma_outputs = s[['SMA4 0', 'SMA4 1', 'SMA4 2']].replace(r'\s*', np.nan, regex=True)

    values_ph = tf.placeholder(tf.float32)
    v = Map(SimpleMovingAverage(4), size=3)
    o, _ = v(values_ph)

    with tf.Session() as sess:
      output = sess.run(o, {
        values_ph: values
      })

    np.testing.assert_almost_equal(output,
      sma_outputs.values, decimal=3)
