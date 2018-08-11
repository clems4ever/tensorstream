import tensorflow as tf

def no_fees(shape):
  def model(quantity, unit_price):
    return tf.zeros(shape, dtype=tf.float32)
  return model

def us_fees(shape):
  def model(quantity, unit_price):
    return tf.constant(0.5, shape=shape) + tf.to_float(quantity) * tf.constant(0.004, shape=shape)
  return model

def fr_fees(shape):
  def model(quantity, unit_price):
    return tf.constant(0.004, shape=shape) * tf.to_float(quantity) * unit_price
  return model

class FeesEvaluator:
  """
  Fees evaluator evaluate per-product fees based on the model given in
  parameters
  """

  def __init__(self, model, shape=()):
    """
    Initialize the evaluator with the per-product fees model
    """
    self.shape = shape
    self.model = model

  def evaluate(self, quantities, prices):
    """
    Evaluation takes a tensor of the quantities to be traded and a tensor of
    the price of each product.
    """

    return tf.where(
      tf.equal(
        quantities,
        tf.zeros(self.shape, dtype=tf.int32)
      ),
      tf.zeros(self.shape, dtype=tf.float32),
      self.model(quantities, prices)
    )
