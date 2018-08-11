import math
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.trading.backtesting.orders_executor import OrdersExecutor

class Backtester(Streamable):
  def __init__(self, capital, strategy_operator, nb_products, nb_inputs,
    fees_model=None):
    super().__init__((
      tf.float32, # Cash
      tf.int32, # Stock quantities
      tf.int32, # Market orders
      tf.float32 # fees per order
    ), ((), (nb_products,), (nb_products,), (nb_products,)))

    self.nb_products = nb_products
    self.strategy = strategy_operator
    self.orders_executor = OrdersExecutor(capital, fees_model, nb_products)
    self.zeros_f = tf.zeros([self.nb_products], dtype=tf.float32)
    self.zeros_i = tf.zeros([self.nb_products], dtype=tf.int32)

    self.initial_state = (
      # Is first iteration? Required to warmup the last_inputs variable
      tf.constant(True),
      # last_inputs, we keep it in the state because we are evaluating the
      # strategy at time t but the orders are exectuted at time t+1
      tuple([tf.fill([nb_products], math.nan)] * nb_inputs),
      self.orders_executor.initial_state,
      self.strategy.initial_state
    )

  def step(self, open_p, close_p, more_inputs,
           is_first_iteration, last_inputs, last_orders_executor_state,
           last_strategy_state):

    safe_open_p = tf.where(tf.is_nan(open_p), self.zeros_f, open_p)
    safe_close_p = tf.where(tf.is_nan(close_p), self.zeros_f, close_p)
    current_inputs = (safe_open_p, safe_close_p, *more_inputs)
    last_cash = last_orders_executor_state[0]
    last_stocks = last_orders_executor_state[1]

    def warmup():
      """
      This function is executed the first time the operator is called.
      It warmups the buffers to have last day and today prices during the
      calculation.
      """
      return (
        last_cash,
        last_stocks,
        self.zeros_i,
        self.zeros_f
      ), (
        tf.constant(False),
        current_inputs,
        last_orders_executor_state,
        last_strategy_state
      )

    def nominal():
      """
      This function is executed after the first operator call. It execute
      the provided strategy to retrieve the orders and pass them to the
      orders executor to update the account.
      """
      # Run the strategy to get the orders.
      new_market_orders, new_strategy_state =\
        self.strategy(*last_inputs, state=last_strategy_state)

      safe_last_close_p = last_inputs[1]

      # Update the portfolio according to orders.
      orders_executor_outputs, new_orders_executor_state =\
        self.orders_executor(safe_open_p, new_market_orders,
          state=last_orders_executor_state)

      return (
        orders_executor_outputs[0],
        orders_executor_outputs[1],
        new_market_orders,
        orders_executor_outputs[2]
      ), (
        tf.constant(False),
        current_inputs,
        new_orders_executor_state,
        new_strategy_state
      )

    # if last_day_prices contains nan, then this is the first iteration
    # and the model requires a warmup
    return tf.cond(is_first_iteration, warmup, nominal)
