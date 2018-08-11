import tensorflow as tf

from tensorstream.operators.backtesting.account_evaluator import AccountEvaluator
from tensorstream.operators.backtesting.fees import FeesEvaluator
from tensorstream.operators import Streamable

class Rebalancer(Streamable):
  """
  This operator has in charge the rebalancing of assets based on weights
  computed by the strategy
  """
  def __init__(self, capital, nb_products, fees_model=None):
    s = (nb_products,)
    super().__init__([tf.float32, tf.int32, tf.int32, tf.float32], [(), s, s, s]) 
    self.fees_evaluator = FeesEvaluator(nb_products, fees_model)

    self.nb_products = nb_products
    self.initial_state = (
      # last_cash is the amount of non invested cash we have in the account
      float(capital),
      # last_stocks is the number of stocks we have in the portfolio
      tf.zeros([nb_products], dtype=tf.int32)
    )

  def __call__(self, last_close, today_open, weights):
    def fn(last_cash, last_stocks):
      bad_inputs = tf.logical_or(
        tf.equal(last_close, 0.0),
        tf.equal(today_open, 0.0)
      )
      zf = tf.zeros([self.nb_products], dtype=tf.float32)
      account_value_yesterday_close = AccountEvaluator.evaluate(last_cash,
        tf.to_float(last_stocks), last_close)
      account_value_today_open = AccountEvaluator.evaluate(last_cash,
        tf.to_float(last_stocks), today_open)

      account_value_t = tf.fill([self.nb_products],
        account_value_yesterday_close)

      # We evaluate the number of stocks to trade for tomorrow
      amount_per_product = account_value_t * weights
      new_stocks = tf.where(bad_inputs,
        last_stocks, tf.to_int32(tf.floor(amount_per_product / last_close)))
      orders = new_stocks - last_stocks

      # The trades are evaluated with open prices
      actual_fees = self.fees_evaluator.evaluate(tf.abs(orders), today_open)
      new_cash = last_cash - tf.reduce_sum(tf.to_float(orders) * today_open) - tf.reduce_sum(actual_fees)
      return (new_cash, new_stocks, orders, actual_fees), (new_cash, new_stocks)
    return fn


