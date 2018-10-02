import tensorflow as tf
import pandas as pd
import numpy as np

from tensorstream.tests import TestCase
from tensorstream.finance.moving_average import SimpleMovingAverage
from tensorstream.finance.moving_average import ExponentialMovingAverage
from tensorstream.finance.moving_average import RollingMovingAverage

class SimpleMovingAveragesSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_averages_sma.ods', __file__))

  def test_simple_moving_average(self):
    sma4 = SimpleMovingAverage(4)
    sma10 = SimpleMovingAverage(10)

    prices = tf.placeholder(tf.float32)
    sma4_ts, _, _ = sma4(prices)
    sma10_ts, _, _ = sma10(prices)
    single_dim_ts = self.sheets['Sheet1']
    
    with tf.Session() as sess:
      output = sess.run([sma4_ts, sma10_ts], 
        { prices: single_dim_ts['Close'] })

    np.testing.assert_almost_equal(output[0],
      single_dim_ts['SMA4'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      single_dim_ts['SMA10'].values, decimal=3)

class ExponentialMovingAverageSpec(TestCase):
  """Test based on http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages"""
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_averages_ema.ods', __file__))

  def test_exponential_moving_average(self):
    s = self.sheets['Sheet1']
    prices = tf.placeholder(tf.float32)

    ema10 = ExponentialMovingAverage(10)
    ema10_ts, _, _ = ema10(prices)
    
    with tf.Session() as sess:
      output = sess.run(ema10_ts, {prices: s['Close']})

    np.testing.assert_almost_equal(output,
      s['EMA10'].values, decimal=3)

class RollingMovingAverageSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_averages_rma.ods', __file__))

  def test_single_dim(self):
    s = self.sheets['Sheet1']
    rma10 = RollingMovingAverage(10)
    values = tf.placeholder(tf.float32)
    rma10_ts, _, _ = rma10(values)
    
    with tf.Session() as sess:
      output = sess.run(rma10_ts, {
        values: s['Close']
      })

    np.testing.assert_almost_equal(output,
      s['RMA10'].values, decimal=3)
