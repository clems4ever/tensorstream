import unittest
import tensorflow as tf

from tensorstream.streamable import Streamable, flatten

class Simple(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value):
    return value * tf.constant(2), ()

class MultiDim(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value):
    return value * tf.constant([2, 3]), ()

class Tuple(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value1, value2):
    return (value1 * 2, value2 * 3), ()

class List(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value1, value2):
    return [value1 * 2, value2 * 3], ()

class Dict(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value):
    return {'val1': value['val1'] * 2, 'val2': value['val2'] * 3}, ()

class Nested(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value):
    x = value['val1'][0]
    y = value['val1'][1]['a'][0]
    z = value['val2']
    return {'val1': (x * 2, {'a': [y * 3]}), 'val2': z * 3}, ()

class WithState(Streamable):
  def __init__(self):
    super().__init__(tf.constant(0))
  def step(self, value, iteration):
    return value * 2, iteration + 2

class Complex(Streamable):
  def __init__(self):
    super().__init__(
      (tf.constant(0), {'b': tf.zeros(2), 'c': tf.constant(5)})
    )
  def step(self, value1, value2, state1, state2):
    new_state2 = {
      'b': state2['b'] + tf.ones(2),
      'c': state2['c'] - 1
    }
    new_value2 = {'a': value2 * 3}
    return (value1 * 2, new_value2), (state1 + 1, new_state2)

class NestedOperator(Streamable):
  def __init__(self):
    super().__init__()
    self.with_state = WithState()
    self.initial_state = self.with_state.initial_state

  def step(self, value, prev_state):
    new_value, next_state = self.with_state(value, prev_state, streamable=False)
    return new_value + 1, next_state

class BadInputStateOperator(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value, iteration):
    return value, ()

class BadOutputOperator(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value):
    return (value, value), ()

class BadOutputStateOperator(Streamable):
  def __init__(self):
    super().__init__()
  def step(self, value):
    return value, value

class StreamableSpec(unittest.TestCase):
  def test_streamable_step(self):
    sim = Simple()
    x = tf.constant(4)
    y, _ = sim(x, streamable=False)

    with tf.Session() as sess:
      output = sess.run(y)
    self.assertEqual(output, 8)

  def test_streamable_simple_value(self):
    sim = Simple()

    x = tf.constant([0, 1, 2, 3, 4, 5])
    y, _ = sim(x)

    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output.tolist(), [0, 2, 4, 6, 8, 10])

  def test_streamable_multidim(self):
    multi_dim = MultiDim()
    x = tf.constant([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y, _ = multi_dim(x)

    with tf.Session() as sess:
      output = sess.run(y)

    self.assertEqual(output.tolist(), [[0, 0], [2, 3], [4, 6], [6, 9], [8, 12], [10, 15]])

  def test_streamable_tuple(self):
    op = Tuple()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    (z0, z1), _ = op(inputs=(x, y))

    with tf.Session() as sess:
      output = sess.run([z0, z1])

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1].tolist(), [3, 6, 9, 12, 15, 18])

  def test_streamable_list(self):
    op_list = List()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    z, _ = op_list(inputs=(x, y))

    with tf.Session() as sess:
      output = sess.run(z)

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1].tolist(), [3, 6, 9, 12, 15, 18])

  def test_streamable_dict(self):
    op = Dict()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    t, _ = op({'val1': x, 'val2': y})

    with tf.Session() as sess:
      output = sess.run(t)

    self.assertEqual(output['val1'].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output['val2'].tolist(), [3, 6, 9, 12, 15, 18])

  def test_streamable_nested(self):
    op = Nested()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6])
    z = tf.constant([2, 3, 4, 5, 6, 7])
    t, _ = op({'val1': (x, {'a': [y]}), 'val2': z})

    with tf.Session() as sess:
      output = sess.run(t)

    self.assertEqual(output['val1'][0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output['val1'][1]['a'][0].tolist(), [3, 6, 9, 12, 15, 18])
    self.assertEqual(output['val2'].tolist(), [6, 9, 12, 15, 18, 21])

  def test_streamable_bad_input_sizes(self):
    op = Tuple()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([1, 2, 3, 4, 5, 6, 7])
    z, _ = op(inputs=(x, y))

    with self.assertRaises(tf.errors.InvalidArgumentError) as context:
      with tf.Session() as sess:
        output = sess.run(z)

  def test_streamable_with_provided_state(self):
    op_with_state = WithState()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    i = tf.constant(5)
    t, j = op_with_state(x, state=i)

    with tf.Session() as sess:
      output = sess.run([t, j])

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1], 17)

  def test_streamable_with_not_provided_state(self):
    op = WithState()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    t, j = op(x)

    with tf.Session() as sess:
      output = sess.run([t, j])

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1], 12)

  def test_streamable_complex(self):
    op = Complex()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    y = tf.constant([0, 1, 2, 3, 4, 5])
    
    z = op(inputs=(x, y))

    with tf.Session() as sess:
      output, state = sess.run(z)

    self.assertEqual(output[0].tolist(), [0, 2, 4, 6, 8, 10])
    self.assertEqual(output[1]['a'].tolist(), [0, 3, 6, 9, 12, 15])
    self.assertEqual(state[0], 6)
    self.assertEqual(state[1]['b'].tolist(), [6, 6])
    self.assertEqual(state[1]['c'].tolist(), -1)

  def test_streamable_nested_operator(self):
    op = NestedOperator()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    z, s = op(x)

    with tf.Session() as sess:
      output = sess.run([z, s])

    self.assertEqual(output[0].tolist(), [1, 3, 5, 7, 9, 11])
    self.assertEqual(output[1], 12)

  def test_streamable_bad_input_state_operator(self):
    op = BadInputStateOperator()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    with self.assertRaises(Exception) as context:
      z, s = op(x, state=tf.constant(0))

    print(str(context.exception))
    self.assertEqual(str(context.exception), "Input state has wrong type. operator.initial_state: (), input_state: <dtype: 'int32'>.")

  def test_streamable_bad_output_state_operator(self):
    op = BadOutputStateOperator()
    x = tf.constant([0, 1, 2, 3, 4, 5])
    with self.assertRaises(Exception) as context:
      z, s = op(x)

    self.assertEqual(str(context.exception), "Output state has wrong type. operator.initial_state: (), output_state: <dtype: 'int32'>.")

class DeducibleMethodsOperator(Streamable):
  def __init__(self):
    super().__init__()

  def initial_state(self, x):
    return tf.zeros(tf.shape(x), dtype=x.dtype)

  def step(self, x, s):
    return x, s + 2

class StreamableDeducibleMethodsSpec(unittest.TestCase):
  def test_streamable_deduce_from_methods(self):
    li = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    lf = [[7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]
    lu = [8, 10, 12, 14, 16, 18]

    xi = tf.placeholder(tf.int32)
    xf = tf.placeholder(tf.float32)
    xu = tf.placeholder(tf.float32)

    op = DeducibleMethodsOperator()

    yi_model, si_model = op(xi)
    yf_model, sf_model = op(xf)
    yu_model, su_model = op(xu)

    self.assertEqual(yi_model.dtype, tf.int32)
    self.assertEqual(si_model.dtype, tf.int32)

    self.assertEqual(yf_model.dtype, tf.float32)
    self.assertEqual(sf_model.dtype, tf.float32)

    self.assertEqual(yu_model.dtype, tf.float32)
    self.assertEqual(su_model.dtype, tf.float32)

    with tf.Session() as sess:
      yi, yf, yu, si, sf, su = sess.run([
        yi_model, yf_model, yu_model,
        si_model, sf_model, su_model
      ], {
        xi: li, xf: lf, xu: lu
      })

    self.assertEqual(yi.tolist(), li)
    self.assertEqual(yf.tolist(), lf)
    self.assertEqual(yu.tolist(), lu)

    self.assertEqual(si.tolist(), [12, 12])
    self.assertEqual(sf.tolist(), [12, 12])
    self.assertEqual(su.tolist(), 12)

class DeducibleLambdasOperator(Streamable):
  def __init__(self):
    super().__init__()
    self.initial_state = lambda x: tf.zeros(tf.shape(x), dtype=x.dtype)

  def step(self, x, s):
    return x, s + 2

class StreamableDeducibleLambdasSpec(unittest.TestCase):
  def test_streamable_deduce_from_lambdas(self):
    li = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    lf = [[7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]
    lu = [8, 10, 12, 14, 16, 18]

    xi = tf.placeholder(tf.int32)
    xf = tf.placeholder(tf.float32)
    xu = tf.placeholder(tf.float32)

    op = DeducibleLambdasOperator()

    yi_model, si_model = op(xi)
    yf_model, sf_model = op(xf)
    yu_model, su_model = op(xu)

    self.assertEqual(yi_model.dtype, tf.int32)
    self.assertEqual(si_model.dtype, tf.int32)

    self.assertEqual(yf_model.dtype, tf.float32)
    self.assertEqual(sf_model.dtype, tf.float32)

    self.assertEqual(yu_model.dtype, tf.float32)
    self.assertEqual(su_model.dtype, tf.float32)

    with tf.Session() as sess:
      yi, yf, yu, si, sf, su = sess.run([
        yi_model, yf_model, yu_model,
        si_model, sf_model, su_model
      ], {
        xi: li, xf: lf, xu: lu
      })

    self.assertEqual(yi.tolist(), li)
    self.assertEqual(yf.tolist(), lf)
    self.assertEqual(yu.tolist(), lu)

    self.assertEqual(si.tolist(), [12, 12])
    self.assertEqual(sf.tolist(), [12, 12])
    self.assertEqual(su.tolist(), 12)
