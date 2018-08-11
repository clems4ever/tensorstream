import tensorflow as tf

from tensorstream.operators.backtesting.fees import FeesEvaluator, no_fees
from tensorstream.operators import Streamable

class OrdersExecutor(Streamable):
  """
  This operator has in charge the holding of portfolio state and order executions
  """

  def __init__(self, capital, fees_model=None, nb_products=1):
    self.input_shape = (nb_products,) if nb_products > 1 else ()
    super().__init__((tf.float32, tf.int32, tf.float32), ((), self.input_shape, self.input_shape))

    if fees_model is None:
      fees_model = no_fees(self.input_shape)
    self.fees_evaluator = FeesEvaluator(fees_model, self.input_shape)
    self.initial_state = (
      # last_cash is the amount of non invested cash we have in the account
      float(capital),
      # last_stocks is the number of stocks we have in the portfolio
      tf.zeros(shape=self.input_shape, dtype=tf.int32)
    )

  def step(self, prices, market_orders,
           last_cash, last_stocks):
    bad_price = tf.equal(prices, tf.zeros(shape=self.input_shape))

    # We evaluate the number of stocks to trade for tomorrow
    # And keep the current number of stocks if we received a bad price.
    new_stocks = tf.where(bad_price,
      last_stocks,
      last_stocks + market_orders)

    # The trades are evaluated with open prices.
    actual_fees = self.fees_evaluator.evaluate(
      tf.abs(market_orders),
      prices)

    new_cash = last_cash - tf.reduce_sum(
      tf.to_float(market_orders) * prices) -\
      tf.reduce_sum(actual_fees)

    return (
      new_cash,
      new_stocks,
      actual_fees
    ), (
      new_cash,
      new_stocks
    )
