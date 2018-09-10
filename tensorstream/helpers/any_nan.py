import functools
import tensorflow as tf

from tensorstream.helpers.map_fn import map_fn
from tensorstream.helpers.flatten import flatten

def tensor_has_nan(t):
  """
  Return True if there is any nan in tensor t, False otherwise.
  """
  return tf.reduce_any(tf.is_nan(t))

def any_nan(inputs):
  """
  Given tensors in parameters, return true if one
  of them has a nan value.
  """

  has_nan_inputs = flatten(
    map_fn(inputs, [inputs], lambda i: tensor_has_nan(i))
  )

  return functools.reduce(
    lambda acc, x: tf.logical_or(acc, x),
    has_nan_inputs,
    tf.constant(False)
  )

