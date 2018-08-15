import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.finance.backtesting.orders_executor import OrdersExecutor
from tensorstream.finance.backtesting.fees import us_fees

from tensorstream.tests import TestCase

class OrdersExecutorSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('orders_executor.ods', __file__))

  def test_single_dim_no_fees(self):
    single_dim_ts = self.sheets['single_dim'].dropna()

    executor = OrdersExecutor(1000)
    prices = tf.placeholder(tf.float32)
    market_orders = tf.placeholder(tf.int32)

    executor_ts, _ = executor((prices, market_orders))
    
    with tf.Session() as sess:
      output = sess.run(executor_ts, {
        prices: single_dim_ts['price'],
        market_orders: single_dim_ts['market_orders']
      })

    np.testing.assert_almost_equal(output[0],
      single_dim_ts['cash'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      single_dim_ts['quantity'].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      single_dim_ts['fees'].values, decimal=3)

  def test_multi_dim_no_fees(self):
    multi_dim_ts = self.sheets['multi_dim'].dropna()

    executor = OrdersExecutor(1000.0, nb_products=2)
    prices_ph = tf.placeholder(tf.float32, shape=[None, 2])
    prices = multi_dim_ts[['price 0', 'price 1']]

    market_orders_ph = tf.placeholder(tf.int32, shape=[None, 2])
    market_orders = multi_dim_ts[['market_orders 0', 'market_orders 1']].astype('int32')

    executor_ts, _ = executor((prices_ph, market_orders_ph))
    
    with tf.Session() as sess:
      output = sess.run(executor_ts, {
        prices_ph: prices,
        market_orders_ph: market_orders
      })

    np.testing.assert_almost_equal(output[0],
      multi_dim_ts['cash'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      multi_dim_ts[['quantity 0', 'quantity 1']].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      multi_dim_ts[['fees 0', 'fees 1']].values, decimal=3)

  def test_multi_dim_with_us_fees(self):
    multi_dim_ts = self.sheets['multi_dim_us_fees'].dropna()

    executor = OrdersExecutor(1000.0, fees_model=us_fees((2,)), nb_products=2)
    prices_ph = tf.placeholder(tf.float32, shape=[None, 2])
    prices = multi_dim_ts[['price 0', 'price 1']]

    market_orders_ph = tf.placeholder(tf.int32, shape=[None, 2])
    market_orders = multi_dim_ts[['market_orders 0', 'market_orders 1']].astype('int32')

    executor_ts, _ = executor((prices_ph, market_orders_ph))
    
    with tf.Session() as sess:
      output = sess.run(executor_ts, {
        prices_ph: prices,
        market_orders_ph: market_orders
      })

    np.testing.assert_almost_equal(output[0],
      multi_dim_ts['cash'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      multi_dim_ts[['quantity 0', 'quantity 1']].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      multi_dim_ts[['fees 0', 'fees 1']].values, decimal=3)
