import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.operators.correlation import Correlation, CrossCorrelation, AutoCorrelation
from tensorstream.operators.tests import TestCase

class CorrelationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('correlation.ods'))

  def test_single_dim(self):
    s = self.sheets['single_dim'].replace(r'\s*', np.nan, regex=True)
    corr5 = Correlation(5)
    returns0 = tf.placeholder(tf.float32)
    returns1 = tf.placeholder(tf.float32)
    corr5_ts, _ = stream_to_tensor(corr5(Stream(returns0), Stream(returns1)))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: s['Returns 0'],
        returns1: s['Returns 1'],
      })

    np.testing.assert_almost_equal(output,
      s['Correlation'].values, decimal=6)

  def test_multiple_dim(self):
    s = self.sheets['multi_dim'].replace(r'\s*', np.nan, regex=True).head(30)
    corr5 = Correlation(5, shape=(2, 2))
    returns0 = tf.placeholder(tf.float32, shape=[None, 2, 2])
    returns1 = tf.placeholder(tf.float32, shape=[None, 2, 2])
    corr5_ts, _ = stream_to_tensor(corr5(Stream(returns0), Stream(returns1)))

    returns_ts = s[['Returns 0', 'Returns 1']]
    expected_outputs_ts = s[['Correlation 0/0', 'Correlation 0/1', 'Correlation 1/0', 'Correlation 1/1']]
    expected_outputs_ts = expected_outputs_ts.values.reshape((30, 2, 2))

    r1 = pd.concat([
      returns_ts['Returns 0'], returns_ts['Returns 0'],
      returns_ts['Returns 1'], returns_ts['Returns 1']],
      axis=1
    )
    r2 = pd.concat([
      returns_ts['Returns 0'], returns_ts['Returns 1'],
      returns_ts['Returns 0'], returns_ts['Returns 1']],
      axis=1
    )

    r1 = r1.values.reshape((30, 2, 2))
    r2 = r2.values.reshape((30, 2, 2))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: r1,
        returns1: r2,
      })

    np.testing.assert_almost_equal(output,
      expected_outputs_ts, decimal=6)

class CrossCorrelationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('cross_correlation.ods'))

  def test_single_dim(self):
    s = self.sheets['single_dim'].replace(r'\s*', np.nan, regex=True)
    corr5 = CrossCorrelation(period=5, lag=2)
    returns0 = tf.placeholder(tf.float32)
    returns1 = tf.placeholder(tf.float32)
    corr5_ts, _ = stream_to_tensor(corr5(Stream(returns0), Stream(returns1)))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: s['Returns 0'],
        returns1: s['Returns 1'],
      })

    np.testing.assert_almost_equal(output,
      s['Correlation'].values, decimal=6)

  def test_multiple_dim(self):
    s = self.sheets['multi_dim'].replace(r'\s*', np.nan, regex=True).head(30)
    corr5 = CrossCorrelation(period=5, lag=2, shape=(2, 2))
    returns0 = tf.placeholder(tf.float32, shape=[None, 2, 2])
    returns1 = tf.placeholder(tf.float32, shape=[None, 2, 2])
    corr5_ts, _ = stream_to_tensor(corr5(Stream(returns0), Stream(returns1)))

    returns_ts = s[['Returns 0', 'Returns 1']]
    expected_outputs_ts = s[['Correlation 0/0', 'Correlation 0/1', 'Correlation 1/0', 'Correlation 1/1']]
    expected_outputs_ts = expected_outputs_ts.values.reshape((30, 2, 2))

    r1 = pd.concat([
      returns_ts['Returns 0'], returns_ts['Returns 0'],
      returns_ts['Returns 1'], returns_ts['Returns 1']],
      axis=1
    )
    r2 = pd.concat([
      returns_ts['Returns 0'], returns_ts['Returns 1'],
      returns_ts['Returns 0'], returns_ts['Returns 1']],
      axis=1
    )

    r1 = r1.values.reshape((30, 2, 2))
    r2 = r2.values.reshape((30, 2, 2))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: r1,
        returns1: r2,
      })

    np.testing.assert_almost_equal(output, expected_outputs_ts, decimal=6)

class AutoCorrelationSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('auto_correlation.ods'))

  def test_single_dim(self):
    s = self.sheets['single_dim'].replace(r'\s*', np.nan, regex=True)
    corr5 = AutoCorrelation(period=5, lag=2)
    returns0 = tf.placeholder(tf.float32)
    corr5_ts, _ = stream_to_tensor(corr5(Stream(returns0)))
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: s['Returns 0'],
      })

    np.testing.assert_almost_equal(output,
      s['Correlation'].values, decimal=6)

  def test_multiple_dim(self):
    s = self.sheets['multi_dim'].replace(r'\s*', np.nan, regex=True).head(30)
    corr5 = AutoCorrelation(period=5, lag=2, shape=(2,))
    returns0 = tf.placeholder(tf.float32, shape=[None, 2])
    corr5_ts, _ = stream_to_tensor(corr5(Stream(returns0)))

    returns_ts = s[['Returns 0', 'Returns 1']]
    expected_outputs_ts = s[['Correlation 0', 'Correlation 1']]
    expected_outputs_ts = expected_outputs_ts.values

    r1 = pd.concat([returns_ts['Returns 0'], returns_ts['Returns 1']],
      axis=1
    )

    r1 = r1.values
    
    with tf.Session() as sess:
      output = sess.run(corr5_ts, {
        returns0: r1,
      })

    np.testing.assert_almost_equal(output, expected_outputs_ts, decimal=6)
