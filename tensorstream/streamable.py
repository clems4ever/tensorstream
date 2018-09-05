import tensorflow as tf
import functools

from tensorstream.helpers.map_fn import map_fn
from tensorstream.helpers.flatten import flatten

class Streamable:
  def __init__(self, dtype=tf.int32, shape=(), initial_state=()):
    self.dtype = dtype
    self.shape = shape
    self.initial_state = initial_state

  @staticmethod
  def check_types_equality(type1, type2, error_message):
    """
    Check the types are equal or throw otherwise.
    """
    if type1 != type2:
      raise Exception(error_message % (str(type1), str(type2)))
    return
     
    
  @staticmethod
  def to_type(values):
    return map_fn(
      values, [values], lambda val: val.dtype if isinstance(val, tf.Tensor) else tf.convert_to_tensor(val).dtype
    )

  def call_streamed(self, inputs_tensors, state):
    inputs_sizes = tuple(map(lambda x: tf.shape(x)[0], flatten(inputs_tensors)))
    size = inputs_sizes[0]
    same_size = functools.reduce(lambda acc, x: tf.logical_and(acc, tf.equal(x, size)),
      inputs_sizes, tf.constant(True))

    assert_same_size = tf.assert_equal(same_size, tf.constant(True),
      data=inputs_sizes, message="inputs have different sizes")

    def cond(i, loop_inputs, loop_state, outputs):
      return i < size

    def loop(i, loop_inputs, loop_state, loop_outputs):
      inputs_i = map_fn(loop_inputs, [loop_inputs], lambda x: x[i])
      if isinstance(loop_inputs, (tuple, list)):
        current_inputs = inputs_i
      else:
        current_inputs = (inputs_i,)

      Streamable.check_types_equality(
        Streamable.to_type(self.initial_state),
        Streamable.to_type(loop_state),
        "Input state has wrong type. operator.initial_state: %s, input_state: %s."
      )

      if isinstance(loop_state, (tuple, list)):
        previous_state = loop_state
      else:
        previous_state= (loop_state,)
        
      outputs_i, next_state = self.step(*current_inputs, *previous_state)

      Streamable.check_types_equality(
        self.dtype,
        Streamable.to_type(outputs_i),
        "Output has wrong type. operator.dtype: %s, output: %s."
      )

      Streamable.check_types_equality(
        Streamable.to_type(self.initial_state),
        Streamable.to_type(next_state),
        "Output state has wrong type. operator.initial_state: %s, output_state: %s."
      )

      new_outputs = map_fn(outputs_i, [outputs_i, loop_outputs], lambda x, y: y.write(i, x))
      map_fn(loop_state, [loop_state, next_state], lambda x, y: y.set_shape(x.get_shape()))
      return (i + 1, loop_inputs, next_state, new_outputs)

    i0 = tf.constant(0)

    outputs = map_fn(self.dtype, [self.dtype, self.shape],
      lambda x, y: tf.TensorArray(dtype=x, element_shape=y, size=size))

    with tf.control_dependencies([assert_same_size]):
      i_f, inputs_f, state_f, outputs_f = tf.while_loop(cond, loop,
        loop_vars=[i0, inputs_tensors, state, outputs], name="stream_loop")
    outputs = map_fn(outputs_f, [outputs_f], lambda o: o.stack())

    return (outputs, state_f)

  def __call__(self, inputs, state=None, streamable=True):
    if state is None:
      state = self.initial_state

    if streamable:
      return self.call_streamed(inputs, state)
    else:
      if isinstance(state, (tuple, list)):
        state_ = state
      else:
        state_ = (state,)
      if isinstance(inputs, (tuple, list)):
        inputs_ = inputs
      else:
        inputs_ = (inputs,)
      return self.step(*inputs_, *state_) 

  def step(self, *inputs):
    raise Exception("Not implemented")

