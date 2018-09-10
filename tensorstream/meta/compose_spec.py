import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.meta.compose import Compose
from tensorstream.tests import TestCase

class Add(Streamable):
  def __init__(self):
    super().__init__()

  def step(self, x, y):
    return x + y, ()

class Square(Streamable):
  def __init__(self):
    super().__init__(tf.constant(0.0))

  def step(self, x, prev_x):
    return prev_x * prev_x, x

class Fork(Streamable):
  def __init__(self):
    super().__init__()

  def step(self, x):
    return (x, x), ()

class ComposeSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('compose.ods', __file__))

  def test_compose_simple(self):
    sheet = self.sheets['Sheet1']

    op = Compose(Square(), Add())
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    op_ts, _ = op((x, y))

    with tf.Session() as sess:
      output = sess.run(op_ts, {
        x: sheet['x'],
        y: sheet['y'],
      })

    np.testing.assert_almost_equal(output,
      sheet['compose'].values, decimal=3)

  def test_compose_multiple_outputs(self):
    sheet = self.sheets['Sheet1']

    op = Compose(Fork(), Square(), Add())
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    op_ts, _ = op((x, y))

    with tf.Session() as sess:
      output = sess.run(op_ts, {
        x: sheet['x'],
        y: sheet['y'],
      })

    assert(len(output) == 2)
    np.testing.assert_almost_equal(output[0],
      sheet['compose'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      sheet['compose'].values, decimal=3)

