import numpy as np
import tensorflow as tf
import pandas as pd

from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.operators.meta import Join, Fork
from tensorstream.operators.moving_average import SimpleMovingAverage, ExponentialMovingAverage
from tensorstream.operators.tests import TestCase

class JoinSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('join.csv', __file__)).astype('float32')

  def test_join_list(self):
    join = Join(SimpleMovingAverage(5), ExponentialMovingAverage(5))
    values_sma = tf.placeholder(tf.float32)
    values_ema = tf.placeholder(tf.float32)

    join_ts, _ = stream_to_tensor(
      join(
        Stream(values_sma),
        Stream(values_ema)
      )
    )

    with tf.Session() as sess:
      output = sess.run(join_ts, {
        values_sma: self.input_ts['Value'],
        values_ema: self.input_ts['Value']
      })

    np.testing.assert_almost_equal(output[0],
      self.input_ts['SMA5'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['EMA5'].values, decimal=3)

  def test_join_list_with_op_generating_multiple_outputs(self):
    join = Join(Fork(2), ExponentialMovingAverage(5))
    values_sma = tf.placeholder(tf.float32)
    values_ema = tf.placeholder(tf.float32)

    join_ts, _ = stream_to_tensor(join(Stream(values_sma), Stream(values_ema)))

    with tf.Session() as sess:
      output = sess.run(join_ts, {
        values_sma: self.input_ts['Value'],
        values_ema: self.input_ts['Value']
      })

    np.testing.assert_almost_equal(output[0][0],
      self.input_ts['Value'].values, decimal=3)
    np.testing.assert_almost_equal(output[0][1],
      self.input_ts['Value'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['EMA5'].values, decimal=3)

  def test_join_multi_dim(self):
    join = Join(SimpleMovingAverage(5, shape=(2,)), ExponentialMovingAverage(5))
    values_sma = tf.placeholder(tf.float32, shape=(None, 2))
    values_ema = tf.placeholder(tf.float32)

    join_ts, _ = stream_to_tensor(join(Stream(values_sma), Stream(values_ema)))

    df = self.input_ts[['Value']].copy()
    df['Value2'] = df['Value']
   
    with tf.Session() as sess:
      output = sess.run(join_ts, {
        values_sma: df.values,
        values_ema: self.input_ts['Value']
      })

    sma5 = self.input_ts[['SMA5']].copy()
    sma5['SMA5_2'] = sma5['SMA5']

    np.testing.assert_almost_equal(output[0],
      sma5.values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['EMA5'].values, decimal=3)

