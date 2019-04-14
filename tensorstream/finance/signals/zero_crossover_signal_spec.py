import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.finance.signals.zero_crossover_signal import ZeroCrossoverSignal
from tensorstream.tests import TestCase


class ZeroCrossoverSignalSpec(TestCase):
    def setUp(self):
        self.sheets = self.read_ods(
            self.from_test_res("zero_crossover_signal.ods", __file__)
        )

    def test_single_dim(self):
        s = self.sheets["Sheet1"]
        trade_signal = ZeroCrossoverSignal()
        histograms = tf.placeholder(tf.float32)
        trade_signal_ts, _, _ = trade_signal(histograms)

        with tf.Session() as sess:
            output = sess.run(trade_signal_ts, {histograms: s["Histogram"]})

        np.testing.assert_almost_equal(output, s["Signals"].values, decimal=3)

