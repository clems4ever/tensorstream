import unittest
import tensorflow as tf
from tensorstream.finance.backtesting.account_evaluator import AccountEvaluator

class AccountEvaluatorSpec(unittest.TestCase):
  def test_account_evaluator(self):
    cash = 100.0
    quantities = [2.0, 0.0, 5.0]
    prices = [23.0, 45.5, 56.2]

    account = {
      'cash': cash,
      'quantities': quantities
    }
    
    with tf.Session() as sess:
      valuation = sess.run(AccountEvaluator.evaluate(cash, quantities, prices))

    self.assertAlmostEqual(valuation, 427.0, delta=0.0001)
