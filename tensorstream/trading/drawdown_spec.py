import numpy as np
import tensorflow as tf

from tensorstream.trading.drawdown import Drawdown, avg
from tensorstream.tests import TestCase

class AverageTrueRangeSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('drawdown.csv', __file__)).astype('float32')

  def test_drawdown(self):
    drawdown = Drawdown()
    values = tf.placeholder(tf.float32)

    drawdown_ts, _ = drawdown(values)

    with tf.Session() as sess:
      output = sess.run(drawdown_ts, {
        values: self.input_ts['Value'],
      })

    np.testing.assert_almost_equal(output[0],
      self.input_ts['Drawdown'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['Time Under Water'].values, decimal=3)

  def test_average_drawdown(self):
    actual_values = [
      100.0, 101.0, 103.0, 102.5, 101.3, 100.5, 101.4, 104.0, 104.5, 103.5,
      102.4, 105.0, 105.5, 107.0, 106.4, 106.3, 106.2, 107.0, 108.0, 108.6,
      108.0, 108.7
    ]
      
    drawdown_op = Drawdown()
    values = tf.placeholder(tf.float32)

    (drawdown, drawdown_days), _ = drawdown_op(values)
    mean_drawdown, mean_drawdown_days = avg(drawdown, drawdown_days)

    with tf.Session() as sess:
      output = sess.run([mean_drawdown, mean_drawdown_days], {
        values: actual_values
      })

    np.testing.assert_almost_equal(output[0], -0.014342278, decimal=3)
    np.testing.assert_almost_equal(output[1], 2.75, decimal=3)
