import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.streamable import stream_to_tensor, Stream, Streamable
from tensorstream.operators.backtesting.backtester import Backtester

from tensorstream.operators.tests import TestCase

class Strategy(Streamable):
  def __init__(self, orders):
    super().__init__(tf.int32, (2,))
    self.orders = tf.constant(orders)
    self.initial_state = 1

  def step(self, open_p, close_p, low_p, high_p, volumes, 
           iteration):
    return self.orders[iteration], iteration + 1

class BacktesterSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_ods(
      self.from_test_res("backtester.ods"))['Sheet1'].astype('float32').dropna()

  def one_product_inputs(self, ticker):
    close_p = self.input_ts['%s close' % ticker]
    open_p = self.input_ts['%s open' % ticker]
    low_p = self.input_ts['%s low' % ticker]
    high_p = self.input_ts['%s high' % ticker]
    volume = self.input_ts['%s volume' % ticker]
    return pd.DataFrame(collections.OrderedDict([
      ('open', open_p),
      ('close', close_p),
      ('low', low_p),
      ('high', high_p),
      ('volume', volume)
    ]))

  def extract_inputs(self, tickers):
    return list(map(lambda ticker: self.one_product_inputs(ticker), tickers))

  def extract_quantities(self, tickers):
    quantities = map(lambda ticker: self.input_ts['%s qty' % ticker], tickers)
    return pd.concat(quantities, axis=1).astype('int32')

  def extract_orders(self, tickers):
    quantities = map(lambda ticker: self.input_ts['%s order' % ticker], tickers)
    return pd.concat(quantities, axis=1).astype('int32')

  def extract_and_join_columns(self, time_series, key):
    return pd.concat(map(lambda ts: ts[key], time_series), axis=1)

  def test_backtester(self):
    tickers = ['MSFT', 'INTC']
    time_series = self.extract_inputs(tickers)
    quantities = self.extract_quantities(tickers)
    orders = self.extract_orders(tickers)

    strategy = Strategy(orders.values)
    backtester = Backtester(1000.0, strategy, nb_products=2, nb_inputs=5)
    close_p = tf.placeholder(tf.float32)
    open_p = tf.placeholder(tf.float32)
    low_p = tf.placeholder(tf.float32)
    high_p = tf.placeholder(tf.float32)
    volumes = tf.placeholder(tf.float32)

    backtester_ts, _ = stream_to_tensor(backtester(
      Stream(open_p),
      Stream(close_p),
      [Stream(low_p), Stream(high_p), Stream(volumes)]
    ))

    with tf.Session() as sess:
      output = sess.run(backtester_ts, {
        close_p: self.extract_and_join_columns(time_series, 'close'),
        low_p:  self.extract_and_join_columns(time_series, 'low'),
        high_p:  self.extract_and_join_columns(time_series, 'high'),
        open_p:  self.extract_and_join_columns(time_series, 'open'),
        volumes:  self.extract_and_join_columns(time_series, 'volume'),
      })

    np.testing.assert_almost_equal(output[0],
      self.input_ts['Cash'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      quantities.values, decimal=3)
