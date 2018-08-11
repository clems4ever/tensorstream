import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.meta import Fork
from tensorstream.meta.compose import Compose
from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.tests import TestCase
from tensorstream.trading.moving_average import SimpleMovingAverage
from tensorstream.trading.returns import Returns

class ComposeSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('compose.csv', __file__)).astype('float32')

  def test_composition(self):
    var1_sma5 = Compose(Returns(1), SimpleMovingAverage(4))
    values = tf.placeholder(tf.float32)
    var1_sma5_ts, _ = stream_to_tensor(var1_sma5(Stream(values)))

    with tf.Session() as sess:
      output = sess.run(var1_sma5_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['Composition'].values, decimal=3)

  def test_composition_multiple_outputs(self):
    var1_sma5 = Compose(Fork(2), Returns(1), SimpleMovingAverage(4))
    values = tf.placeholder(tf.float32)
    var1_sma5_ts, _ = stream_to_tensor(var1_sma5(Stream(values)))

    with tf.Session() as sess:
      output = sess.run(var1_sma5_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output[0],
      self.input_ts['Composition'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['Composition'].values, decimal=3)
