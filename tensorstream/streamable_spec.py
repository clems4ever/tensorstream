import unittest
import tensorflow as tf

from tensorstream.streamable import Streamable, flatten

class Simple(Streamable):
  def step(self, value):
    return value * tf.constant(2), (), ()

class SimpleSpec(unittest.TestCase):
  def test_simple_step(self):
    sim = Simple()
    x = tf.constant(4)
    y, _, _ = sim(x, streamable=False)

    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output, 8)

  def test_streamable_simple_value(self):
    sim = Simple()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y, _, _ = sim(x)

    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output.tolist(), [0, 2, 4, 6, 8, 10])

class MultiDim(Streamable):
  def step(self, value):
    return value * tf.constant([2, 3]), (), ()

class MultiDimSpec(unittest.TestCase):
  def test_streamable_multidim(self):
    multi_dim = MultiDim()
    x = tf.constant([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y, _, _ = multi_dim(x)

    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output.tolist(), [[0, 0], [2, 3], [4, 6], [6, 9], [8, 12], [10, 15]])

class Tuple(Streamable):
  def step(self, value1, value2):
    return (value1 * 2, value2 * 3), (), ()

class TupleSpec(unittest.TestCase):
  def test_streamable_tuple(self):
    op = Tuple()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    (z0, z1), _, _ = op(inputs=(x, y))

    with tf.Session() as sess:
      output = sess.run([z0, z1])

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1].tolist(), [3, 6, 9, 12, 15, 18])

class List(Streamable):
  def step(self, value1, value2):
    return [value1 * 2, value2 * 3], (), ()

class ListSpec(unittest.TestCase):
  def test_streamable_list(self):
    op_list = List()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    z, _, _ = op_list(inputs=(x, y))

    with tf.Session() as sess:
      output = sess.run(z)

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1].tolist(), [3, 6, 9, 12, 15, 18])

class Dict(Streamable):
  def step(self, value):
    return {'val1': value['val1'] * 2, 'val2': value['val2'] * 3}, (), ()

class DictSpec(unittest.TestCase):
  def test_streamable_dict(self):
    op = Dict()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    t, _, _ = op({'val1': x, 'val2': y})

    with tf.Session() as sess:
      output = sess.run(t)

    self.assertEqual(output['val1'].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output['val2'].tolist(), [3, 6, 9, 12, 15, 18])

class Nested(Streamable):
  def step(self, value):
    x = value['val1'][0]
    y = value['val1'][1]['a'][0]
    z = value['val2']
    return {'val1': (x * 2, {'a': [y * 3]}), 'val2': z * 3}, (), ()

class NestedSpec(unittest.TestCase):
  def test_streamable_nested(self):
    op = Nested()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    z = tf.constant([2, 3, 4, 5, 6, 7])
    t, _, _ = op({'val1': (x, {'a': [y]}), 'val2': z})

    with tf.Session() as sess:
      output = sess.run(t)

    self.assertEqual(output['val1'][0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output['val1'][1]['a'][0].tolist(), [3, 6, 9, 12, 15, 18])
    self.assertEqual(output['val2'].tolist(), [6, 9, 12, 15, 18, 21])

class WithState(Streamable):
  def step(self, value, iteration=None):
    if iteration is None:
      iteration = tf.zeros(dtype=value.dtype, shape=tf.shape(value))
    return value * 2, iteration + 2, iteration

class WithStateSpec(unittest.TestCase):
  def test_streamable_step_with_state_not_provided(self):
    ws = WithState()
    x = tf.constant(4)
    y = ws(x, streamable=False)
    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output[0], 8)
    self.assertEqual(output[1], 2)

  def test_streamable_step_with_state_provided(self):
    ws = WithState()
    x = tf.constant(4)
    y = ws(x, state=tf.constant(5), streamable=False)
    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output[0], 8)
    self.assertEqual(output[1], 7)

  def test_streamable_with_provided_state(self):
    op_with_state = WithState()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    i = tf.constant(5)
    t, j, _ = op_with_state(x, state=i)

    with tf.Session() as sess:
      output = sess.run([t, j])

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1], 17)

  def test_streamable_with_not_provided_state(self):
    op = WithState()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    t, j, _ = op(x)

    with tf.Session() as sess:
      output = sess.run([t, j])

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1], 12)

class Complex(Streamable):
  def step(self, value1, value2, state1=None, state2=None):
    if state1 is None:
      state1 = tf.constant(0)
    if state2 is None:
      state2 = {'b': tf.zeros(2), 'c': tf.constant(5)}

    new_state2 = {
      'b': state2['b'] + tf.ones(2),
      'c': state2['c'] - 1
    }
    new_value2 = {'a': value2 * 3}

    return (value1 * 2, new_value2), (state1 + 1, new_state2), (state1, state2)

class ComplexSpec(unittest.TestCase):
  def test_streamable_complex(self):
    op = Complex()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([0, 1, 2, 3, 4, 5])
    
    z = op(inputs=(x, y))

    with tf.Session() as sess:
      output, state, init_state = sess.run(z)

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1]['a'].tolist(), [0, 3, 6, 9, 12, 15])
    self.assertEqual(state[0], 6)
    self.assertEqual(state[1]['b'].tolist(), [6, 6])
    self.assertEqual(state[1]['c'].tolist(), -1)

class NestedOperator(Streamable):
  def __init__(self):
    super().__init__()
    self.with_state = WithState()

  def step(self, value, prev_state=None):
    new_value, next_state, init_state = self.with_state(
      value, prev_state, streamable=False) 
    return new_value + 1, next_state, init_state

class NestedOpSpec(unittest.TestCase):
  def test_streamable_nested_operator(self):
    op = NestedOperator()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    z, s, _ = op(x)

    with tf.Session() as sess:
      output = sess.run([z, s])

    self.assertEqual(output[0].tolist(), [1, 3, 5, 7, 9, 11])
    self.assertEqual(output[1], 12)

class MultiNestedOperators(Streamable):
  def __init__(self):
    self.with_state = WithState()
  def step(self, value, iteration=None, prev_state1=None, prev_state2=None):
    if iteration is None:
      iteration = tf.constant(0)

    new_value1, next_state1, init_state1 = self.with_state(value, prev_state1, streamable=False)
    new_value2, next_state2, init_state2 = self.with_state(new_value1, prev_state2, streamable=False)
    return new_value2 + 1, (iteration + 1, next_state1, next_state2), (iteration, init_state1, init_state2)

class MultiNestedOperatorsSpec(unittest.TestCase):
  def test_streamable_multi_nested_operators(self):
    op = MultiNestedOperators()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    z, s, _ = op(x)

    with tf.Session() as sess:
      output = sess.run([z, s])

    self.assertEqual(output[0].tolist(), [1, 5, 9, 13, 17, 21])
    self.assertEqual(output[1], (6, 12, 12))

class BadInputStateOperator(Streamable):
  def step(self, value, iteration=None):
    return value, (), tf.constant(0)

class BadOutputStateOperator(Streamable):
  def step(self, value):
    return value, value, ()

class StreamableSpec(unittest.TestCase):
  def test_streamable_bad_input_sizes(self):
    op = Tuple()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6, 7])
    z, _, _ = op(inputs=(x, y))

    with self.assertRaises(tf.errors.InvalidArgumentError) as context:
      with tf.Session() as sess:
        output = sess.run(z)

  def test_streamable_bad_input_state_operator(self):
    op = BadInputStateOperator()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    with self.assertRaises(Exception) as context:
      z, s, _ = op(x, state=tf.constant(0))

    self.assertEqual(str(context.exception), "Provided state and model state has different types (<dtype: 'int32'> != ()).")

  def test_streamable_bad_output_state_operator(self):
    op = BadOutputStateOperator()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    with self.assertRaises(Exception) as context:
      z, s, _ = op(x)

    self.assertEqual(str(context.exception), "Provided state and model state has different types (() != <dtype: 'int32'>).")

