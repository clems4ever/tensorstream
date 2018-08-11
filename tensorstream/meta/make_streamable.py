import tensorflow as tf
from tensorstream.streamable import Streamable

class LambdaStreamable(Streamable):
  def __init__(self, fn, dtype=tf.int32, shape=()):
    super().__init__(dtype, shape)
    self.fn = fn

  def step(self, *inputs):
    output = self.fn(*inputs)
    return output, ()

def make_streamable(fn, dtype=tf.int32, shape=()):
  return LambdaStreamable(fn, dtype, shape)
