import numpy as np
import tensorflow as tf
import pandas as pd

from tensorstream.meta import Join
from tensorstream.streamable import Streamable
from tensorstream.tests import TestCase

class Add(Streamable):
  def properties(self, x, y):
    return x, ()
  def step(self, x, y):
    return x + y, ()

class Square(Streamable):
  def properties(self, x):
    return x, tf.constant(0.0)
  def step(self, x, prev_x):
    return prev_x * prev_x, x

class Fork(Streamable):
  def properties(self, x):
    return (x, x), ()
  def step(self, x):
    return (x, x), ()

class JoinSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('join.ods', __file__))

  def test_join_simple(self):
    sheet = self.sheets['Sheet1']
    join = Join(Add(), Square())
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = tf.placeholder(tf.float32)

    join_ts, _ = join(((x, y), z))

    with tf.Session() as sess:
      output = sess.run(join_ts, {
        x: sheet['x'],
        y: sheet['y'],
        z: sheet['z']
      })

    np.testing.assert_almost_equal(output[0],
      sheet['add'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      sheet['square'].values, decimal=3)

  def test_join_multi_outputs(self):
    sheet = self.sheets['Sheet2']
    join = Join(Fork(), Square())
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    join_ts, _ = join((x, y))

    with tf.Session() as sess:
      output = sess.run(join_ts, {
        x: sheet['x'],
        y: sheet['y']
      })

    np.testing.assert_almost_equal(output[0][0],
      sheet['fork'], decimal=3)
    np.testing.assert_almost_equal(output[0][1],
      sheet['fork'], decimal=3)
    np.testing.assert_almost_equal(output[1],
      sheet['square'], decimal=3)

