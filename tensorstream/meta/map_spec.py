import numpy as np
import tensorflow as tf

from tensorstream.meta.ffill import FFill
from tensorstream.meta.map import Map
from tensorstream.streamable import Streamable
from tensorstream.tests import TestCase

class Square(Streamable):
  def __init__(self):
    super().__init__(tf.constant(0.0))

  def step(self, x, prev_x):
    return prev_x * prev_x, x

class Fork(Streamable):
  def __init__(self):
    super().__init__(tf.constant(0.0))

  def step(self, x, prev_x):
    return (prev_x, prev_x), x

class StateMultiDim(Streamable):
  def __init__(self):
    super().__init__()
    
  def initial_state(self, x):
    return tf.zeros([2])
    
  def step(self, x, prev_x):
    return prev_x, tf.fill([2], x)

class MapSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('map.ods', __file__))

  def test_map_simple(self):
    sheet = self.sheets['Sheet1']
    x = tf.placeholder(tf.float32)
    v = Map(Square(), size=3)
    o, _ = v(x)

    with tf.Session() as sess:
      output = sess.run(o, {
        x: sheet[['x0', 'x1', 'x2']]
      })

    expected = sheet[['s0', 's1', 's2']]

    np.testing.assert_almost_equal(output,
      expected.values, decimal=3)

  def test_map_multi_output(self):
    sheet = self.sheets['Sheet2']
    x = tf.placeholder(tf.float32)
    v = Map(Fork(), size=3)
    o, _ = v(x)

    with tf.Session() as sess:
      output = sess.run(o, {
        x: sheet[['x0', 'x1', 'x2']]
      })

    assert(len(output) == 2)
    np.testing.assert_almost_equal(output[0],
      sheet[['y0', 'y1', 'y2']].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      sheet[['y0', 'y1', 'y2']].values, decimal=3)

  def test_map_multi_dim(self):
    sheet = self.sheets['Sheet3']
    x = tf.placeholder(tf.float32)
    v = Map(StateMultiDim(), size=3)
    o, _ = v(x)

    with tf.Session() as sess:
      output = sess.run(o, {
        x: sheet[['x0', 'x1', 'x2']]
      })

    expected = sheet[['y0', 'y1', 'y2', 'y3', 'y4', 'y5']].values
    expected = np.reshape(expected, (12, 3, 2))
    np.testing.assert_almost_equal(output,
      expected, decimal=3)

class MapFFillSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('map.ods', __file__))

  def test_map_ffill(self):
    sheet = self.sheets['map_ffill']
    values = sheet[['x0', 'x1', 'x2']]

    values_ph = tf.placeholder(tf.float32)
    v = Map(FFill(Square()), size=3)
    o, _ = v(values_ph)

    with tf.Session() as sess:
      output = sess.run(o, {
        values_ph: values
      })

    expected = sheet[['y0', 'y1', 'y2']]
    np.testing.assert_almost_equal(output,
      expected, decimal=3)
