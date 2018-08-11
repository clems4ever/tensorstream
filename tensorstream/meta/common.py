import tensorflow as tf
from tensorstream.meta.make_streamable import make_streamable

# Subtract
def Sub(dtype=tf.float32, shape=()):
  return make_streamable(lambda x, y: x - y, dtype, shape)

# Add
def Add(dtype=tf.float32, shape=()):
  return make_streamable(lambda x, y: x + y, dtype, shape)

# Multiply
def Mul(dtype=tf.float32, shape=()):
  return make_streamable(lambda x, y: x * y, dtype, shape)

# Fork series
def Fork(count, dtype=tf.float32, shape=()):
  return make_streamable(lambda x: tuple([x] * count), tuple([dtype] * count), tuple([shape] * count))

# Identity
def Identity(dtype=tf.float32, shape=()):
  return make_streamable(lambda x: x, dtype, shape)

def Select(indices, dtype=tf.float32, shape=()):
  if isinstance(indices, tuple):
    return make_streamable(lambda *values: (values[i] for i in indices), dtype, shape)
  elif isinstance(indices, list):
    return make_streamable(lambda *values: [values[i] for i in indices], dtype, shape)
  else:
    return make_streamable(lambda *values: values[indices], dtype, shape)

# Positive
def Positive(dtype=tf.float32, shape=()):
  return make_streamable(lambda x: tf.where(
    tf.greater(x, tf.zeros(dtype=dtype, shape=shape)),
    tf.ones(dtype=dtype, shape=shape),
    tf.zeros(dtype=dtype, shape=shape)), 
  dtype, shape)

# Negative
def Negative(dtype=tf.float32, shape=()):
  return make_streamable(lambda x: tf.where(
    tf.less(x, tf.zeros(dtype=dtype, shape=shape)),
    tf.ones(dtype=dtype, shape=shape),
    tf.zeros(dtype=dtype, shape=shape)),
  dtype, shape)
