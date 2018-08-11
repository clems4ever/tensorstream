import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.trading.moving_standard_deviation import MovingStandardDeviation

class MovingStandardDeviationSpec(TestCase):
  # Data from http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('moving_standard_deviation.csv', __file__))\
        .fillna(0.0).astype('float32')

  def test_global_min(self):
    volatility = MovingStandardDeviation(10)
    values = tf.placeholder(tf.float32)
    volatility_ts, _ = volatility(values)

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['Volatility'].values, decimal=3)
