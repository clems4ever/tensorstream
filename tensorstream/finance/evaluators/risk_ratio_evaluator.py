import math
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation

class RiskRatioEvaluator(Streamable):
    def __init__(self, stop_factor, limit_factor, volatility_period=5,
      max_bet_duration=None, dtype=tf.float32, shape=()):

      super().__init__((tf.int32, tf.int32, dtype, tf.int32), tuple([shape]*4))
      self.msd_op = MovingStandardDeviation(period=volatility_period)
      self.initial_state = (
          tf.constant(math.nan), # stop price
          tf.constant(math.nan), # limit price
          tf.constant(math.nan), # buy price
          tf.constant(0), # bet_duration
          tf.constant(0), # successful bets
          tf.constant(0), # total bets
          self.msd_op.initial_state
      )
      self.stop_factor = stop_factor
      self.limit_factor = limit_factor
      self.max_bet_duration = max_bet_duration
        
    def step(self, low, high, close, signal,
      previous_stop_price,
      previous_limit_price,
      previous_buy_price,
      previous_bet_duration,
      previous_successful_bets,
      previous_total_bets,
      previous_volatility_state): 
      
      volatility, next_volatility_state = self.msd_op(
        inputs=close,
        state=previous_volatility_state,
        streamable=False
      )
      
      is_nan = tf.is_nan(previous_stop_price)

      stop_hit = tf.logical_and(
        tf.logical_not(is_nan),
        tf.less_equal(low, previous_stop_price)
      )
      limit_hit = tf.logical_and(
        tf.logical_not(is_nan),
        tf.greater_equal(high, previous_limit_price)
      )

      if not self.max_bet_duration is None:
        max_bet_duration_reached = tf.logical_and(
          tf.logical_not(is_nan),
          tf.equal(previous_bet_duration, self.max_bet_duration)
        )

      buy_signal = tf.logical_and(
        is_nan,
        tf.equal(signal, 1)
      )
      
      def buy():
        stop_price = close - self.stop_factor * volatility
        limit_price = close + self.limit_factor * volatility
        return (
          stop_price,
          limit_price,
          close,
          1,
          previous_successful_bets,
          previous_total_bets,
          1
        )
      
      def wait():
        next_bet_duration = previous_bet_duration + 1
        return (
          previous_stop_price,
          previous_limit_price,
          previous_buy_price,
          next_bet_duration,
          previous_successful_bets,
          previous_total_bets,
          0
        )
      
      def sell_after_stop_hit():
        return (
          tf.constant(math.nan),
          tf.constant(math.nan),
          tf.constant(math.nan),
          0,
          previous_successful_bets,
          previous_total_bets + 1,
          -1
        )

      def sell_after_limit_hit():
        return (
          tf.constant(math.nan),
          tf.constant(math.nan),
          tf.constant(math.nan),
          0,
          previous_successful_bets + 1,
          previous_total_bets + 1,
          -1
        )

      def sell_after_max_bet_duration_reached():
        next_successful_bets = tf.where(
          tf.greater(close, previous_buy_price),
          previous_successful_bets + 1,
          previous_successful_bets
        )
        return (
          tf.constant(math.nan),
          tf.constant(math.nan),
          tf.constant(math.nan),
          0,
          next_successful_bets,
          previous_total_bets + 1,
          -1
       )

      cases = [
        (stop_hit, sell_after_stop_hit),
        (limit_hit, sell_after_limit_hit),
        (buy_signal, buy)
      ]

      if not self.max_bet_duration is None:
        cases.append((max_bet_duration_reached, sell_after_max_bet_duration_reached))
      
      next_state  = tf.case(cases, default=wait, exclusive=False)

      return (
        next_state[4],
        next_state[5],
        volatility,
        next_state[6]
      ), (
        *next_state[0:6],
        next_volatility_state
      )
