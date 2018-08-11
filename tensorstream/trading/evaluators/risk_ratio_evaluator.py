import math
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.trading import Volatility

class RiskRatioEvaluator(Streamable):
    def __init__(self, stop_factor, limit_factor, volatility_period=5, dtype=tf.float32, shape=()):
        super().__init__((tf.int32, tf.int32, dtype, tf.int32), tuple([shape]*4))
        self.volatility_op = Volatility(period=volatility_period)
        self.initial_state = (
            tf.constant(math.nan), # stop price
            tf.constant(math.nan), # limit price
            tf.constant(0), # successful bets
            tf.constant(0), # total bets
            self.volatility_op.initial_state
        )
        self.stop_factor = stop_factor
        self.limit_factor = limit_factor
        
    def step(self, low, high, close, signal,
      previous_stop_price,
      previous_limit_price,
      previous_successful_bets,
      previous_total_bets,
      previous_volatility_state): 
      
      volatility, next_volatility_state = self.volatility_op(close, state=previous_volatility_state)
      
      is_nan = tf.is_nan(previous_stop_price)
      stop_hit = tf.logical_and(
          tf.logical_not(is_nan),
          tf.less_equal(low, previous_stop_price)
      )
      limit_hit = tf.logical_and(
          tf.logical_not(is_nan),
          tf.greater_equal(high, previous_limit_price)
      )
      buy_signal = tf.logical_and(
          is_nan,
          tf.equal(signal, 1)
      )
      
      def buy():
          stop_price = close - self.stop_factor * volatility
          limit_price = close + self.limit_factor * volatility
          return stop_price, limit_price, previous_successful_bets, previous_total_bets, 1
      
      def wait():
          return previous_stop_price, previous_limit_price, previous_successful_bets, previous_total_bets, 0
      
      def buy_or_wait():
          return tf.cond(buy_signal, buy, wait)
      
      stop_hit_fn = lambda: (tf.constant(math.nan), tf.constant(math.nan), previous_successful_bets, previous_total_bets + 1, -1)
      limit_hit_fn = lambda: (tf.constant(math.nan), tf.constant(math.nan), previous_successful_bets + 1, previous_total_bets + 1, -1)

      next_stop_price, next_limit_price, next_successful_bets, next_total_bets, next_signal = tf.case([
          (stop_hit, stop_hit_fn),
          (limit_hit, limit_hit_fn)
      ], default=buy_or_wait, exclusive=False)
      
      
      return (next_successful_bets, next_total_bets, volatility, next_signal), (
          next_stop_price,
          next_limit_price,
          next_successful_bets,
          next_total_bets,
          next_volatility_state
      )
