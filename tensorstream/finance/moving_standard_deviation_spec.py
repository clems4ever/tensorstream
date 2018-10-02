import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation

class MovingStandardDeviationSpec(TestCase):
  # Data from http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_standard_deviation.ods', __file__))

  def test_moving_standard_deviation_single_dim(self):
    volatility = MovingStandardDeviation(10)
    values = tf.placeholder(tf.float32)
    volatility_ts, _, _ = volatility(values)

    inputs = self.sheets['Sheet1']

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: inputs['Value'] })

    np.testing.assert_almost_equal(output,
      inputs['mstdev'].values, decimal=3)

