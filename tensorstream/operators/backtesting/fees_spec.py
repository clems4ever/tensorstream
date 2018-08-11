import unittest
import numpy as np
import tensorflow as tf

from tensorstream.operators.backtesting.fees import FeesEvaluator, no_fees

class FeesSpec(unittest.TestCase):
  def test_fees_evaluator_with_model(self):
    shape = [3]
    model = lambda quantity, price: tf.constant(0.5, shape=shape) + tf.to_float(quantity) * price * tf.constant(0.004, shape=shape)
    evaluator = FeesEvaluator(model, shape)

    r = evaluator.evaluate(
      tf.constant([20, 10, 0]), tf.constant([32.5, 14.9, 20.3]))
    with tf.Session() as sess:
      fees = sess.run(r)
    
    expected_fees = [3.1, 1.096, 0.0]
    np.testing.assert_almost_equal(fees, expected_fees, decimal=3)

  def test_fees_evaluator_without_model(self):
    shape = [3]
    evaluator = FeesEvaluator(no_fees(shape), shape)

    r = evaluator.evaluate(
      tf.constant([20, 10, 0]), tf.constant([32.5, 14.9, 20.3]))
    with tf.Session() as sess:
      fees = sess.run(r)
    
    expected_fees = [0.0, 0.0, 0.0]
    np.testing.assert_almost_equal(fees, expected_fees, decimal=3)
