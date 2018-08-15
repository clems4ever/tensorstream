import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.finance.signals.zero_crossover_signal import ZeroCrossoverSignal
from tensorstream.tests import TestCase

class ZeroCrossoverSignalSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('zero_crossover_signal.ods', __file__))

  def test_single_dim(self):
    s = self.sheets['single_dim']
    trade_signal = ZeroCrossoverSignal()
    histograms = tf.placeholder(tf.float32)
    trade_signal_ts, _ = trade_signal(histograms)

    with tf.Session() as sess:
      output = sess.run(trade_signal_ts, {
        histograms: s['Histogram']
      })

    np.testing.assert_almost_equal(output,
      s['Signals'].values, decimal=3)

  def test_multi_dim(self):
    s = self.sheets['multi_dim'].dropna()
    trade_signal = ZeroCrossoverSignal(shape=(2,))
    histograms = tf.placeholder(tf.float32, shape=[None, 2])
    trade_signal_ts, _ = trade_signal(histograms)

    histograms_ts = s[['Histogram 0', 'Histogram 1']]
    signals_ts = s[['Signals 0', 'Signals 1']]

    with tf.Session() as sess:
      output = sess.run(trade_signal_ts, {
        histograms: histograms_ts
      })

    np.testing.assert_almost_equal(output, signals_ts.values, decimal=3)

