import functools
import tensorflow as tf

from tensorstream.helpers.map_fn import map_fn
from tensorstream.helpers.flatten import flatten

def tensor_has_value(t, value):
  return tf.reduce_any(
    tf.equal(t, tf.fill(tf.shape(t), value))
  )

def any_value(inputs, value):
  """
  Given tensors in parameters, returns true if one
  of them contains the given value.
  """
  has_value_inputs = flatten(
    map_fn(inputs, [inputs], lambda i: tensor_has_value(i, value))
  )

  return functools.reduce(
    lambda acc, x: tf.logical_or(acc, x),
    has_value_inputs,
    tf.constant(False)
  )
