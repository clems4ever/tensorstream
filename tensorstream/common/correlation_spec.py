import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.common.correlation import Correlation, CrossCorrelation, AutoCorrelation
from tensorstream.tests import TestCase

class SimpleCorrelationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('correlation.ods', __file__))

  def test_simple_correlation(self):
    s = self.sheets['Sheet1'].replace(r'\s*', np.nan, regex=True)

    corr5 = Correlation(5)
    returns0 = tf.placeholder(tf.float32)
    returns1 = tf.placeholder(tf.float32)
    corr5_ts, _ = corr5(inputs=(returns0, returns1))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: s['Returns 0'],
        returns1: s['Returns 1'],
      })

    np.testing.assert_almost_equal(output,
      s['Correlation'].values, decimal=6)

class CrossCorrelationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('cross_correlation.ods', __file__))

  def test_cross_correlation(self):
    s = self.sheets['Sheet1'].replace(r'\s*', np.nan, regex=True)
    corr5 = CrossCorrelation(period=5, lag=2)
    returns0 = tf.placeholder(tf.float32)
    returns1 = tf.placeholder(tf.float32)
    corr5_ts, _ = corr5(inputs=(returns0, returns1))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: s['Returns 0'],
        returns1: s['Returns 1'],
      })

    np.testing.assert_almost_equal(output,
      s['Correlation'].values, decimal=6)

class AutoCorrelationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('auto_correlation.ods', __file__))

  def test_auto_correlation(self):
    s = self.sheets['Sheet1'].replace(r'\s*', np.nan, regex=True)
    corr5 = AutoCorrelation(period=5, lag=2)
    returns0 = tf.placeholder(tf.float32)
    corr5_ts, _ = corr5(returns0)
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: s['Returns 0'],
      })

    np.testing.assert_almost_equal(output,
      s['Correlation'].values, decimal=6)
