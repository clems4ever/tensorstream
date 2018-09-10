import tensorflow as tf

from tensorstream.common.make_streamable import make_streamable

# Subtract
def Sub():
  return make_streamable(lambda x, y: x - y)

# Add
def Add():
  return make_streamable(lambda x, y: x + y)

# Multiply
def Mul():
  return make_streamable(lambda x, y: x * y)

# Fork series
def Fork(count):
  return make_streamable(lambda x: tuple([x] * count))

# Identity
def Identity():
  return make_streamable(lambda x: x)

def Select(indices):
  if isinstance(indices, tuple):
    return make_streamable(lambda *values: (values[i] for i in indices))
  elif isinstance(indices, list):
    return make_streamable(lambda *values: [values[i] for i in indices])
  else:
    return make_streamable(lambda *values: values[indices])

# Positive
def Positive():
  return make_streamable(lambda x: tf.where(
    tf.greater(x, tf.zeros(dtype=x.dtype, shape=x.shape)),
    tf.ones(dtype=x.dtype, shape=x.shape),
    tf.zeros(dtype=x.dtype, shape=x.shape))
  )

# Negative
def Negative():
  return make_streamable(lambda x: tf.where(
    tf.less(x, tf.zeros(dtype=x.dtype, shape=x.shape)),
    tf.ones(dtype=x.dtype, shape=x.shape),
    tf.zeros(dtype=x.dtype, shape=x.shape))
  )
