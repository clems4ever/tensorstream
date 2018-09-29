import tensorflow as tf

class Placeholder:
  def __init__(self, dtype, shape=None):
    self.dtype = dtype
    self.shape = shape

  @staticmethod
  def from_tensor(tensor):
    return Placeholder(tensor.dtype, tf.shape(tensor))
