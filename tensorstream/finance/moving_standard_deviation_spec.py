import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation

class MovingStandardDeviationSpec(TestCase):
  # Data from http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_standard_deviation.ods', __file__))

  def test_volatility(self):
    volatility = MovingStandardDeviation(10)
    values = tf.placeholder(tf.float32)
    volatility_ts, _ = volatility(values)

    inputs = self.sheets['single_dim']

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: inputs['Value'] })

    np.testing.assert_almost_equal(output,
      inputs['Volatility'].values, decimal=3)

  def test_volatility_multidim(self):
    volatility = MovingStandardDeviation(10, shape=(2,))
    values = tf.placeholder(tf.float32, shape=(None, 2))
    volatility_ts, _ = volatility(values)

    inputs = self.sheets['multi_dim']
    data = inputs[['Value 1', 'Value 2']]
    expected = inputs[['Volatility 1', 'Volatility 2']]

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: data })

    np.testing.assert_almost_equal(output,
      expected.values, decimal=3)
