import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.meta import Fork, make_streamable
from tensorstream.meta.compose import Compose
from tensorstream.tests import TestCase
from tensorstream.finance.moving_average import SimpleMovingAverage
from tensorstream.finance.returns import Return

class ComposeSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('compose.csv', __file__)).astype('float32')

  def test_composition(self):
    var1_sma5 = Compose(Return(1), SimpleMovingAverage(4))
    values = tf.placeholder(tf.float32)
    var1_sma5_ts, _ = var1_sma5(inputs=values)

    with tf.Session() as sess:
      output = sess.run(var1_sma5_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['Composition'].values, decimal=3)

  def test_composition_multiple_outputs(self):
    var1_sma5 = Compose(Fork(2), Return(1), SimpleMovingAverage(4))
    values = tf.placeholder(tf.float32)
    var1_sma5_ts, _ = var1_sma5(values)

    with tf.Session() as sess:
      output = sess.run(var1_sma5_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output[0],
      self.input_ts['Composition'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['Composition'].values, decimal=3)

  def test_composition_multiple_inputs(self):
    add = Compose(make_streamable(lambda x, y: x + y, dtype=tf.float32))
    values = tf.placeholder(tf.float32)
    add_ts, _ = add((values, values))

    with tf.Session() as sess:
      output = sess.run(add_ts, {
        values: self.input_ts['Value']
      })

    np.testing.assert_almost_equal(output,
      self.input_ts['Value'].values * 2, decimal=3)
