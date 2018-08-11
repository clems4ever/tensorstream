import tensorflow as tf

class AccountEvaluator(object):
  @staticmethod
  def evaluate(cash, quantities, prices):
    return cash + tf.reduce_sum(
      tf.transpose(tf.to_float(quantities)) * prices)
