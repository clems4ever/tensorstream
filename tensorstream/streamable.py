import tensorflow as tf
import functools

from tensorstream.helpers.map_fn import map_fn
from tensorstream.helpers.flatten import flatten

class Streamable:
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

  def call_streamed(self, inputs_tensors, provided_state):
    """
    Stream inputs_tensors into the operator. The variable provided_state can
    be None if initial state is not provided by the user or it can be a compatible
    initial state provided by the user.
    """

    inputs_sizes = tuple(map(lambda x: tf.shape(x)[0], flatten(inputs_tensors)))
    size = inputs_sizes[0]
    same_size = functools.reduce(lambda acc, x: tf.logical_and(acc, tf.equal(x, size)),
      inputs_sizes, tf.constant(True))

    assert_same_size = tf.assert_equal(same_size, tf.constant(True),
      data=inputs_sizes, message="inputs have different sizes.")

    inputs_0 = map_fn(inputs_tensors, [inputs_tensors], lambda x: x[0])
    outputs_0, state, initial_state = self.forward_step(inputs_0, provided_state)

    # if state is provided, we take it as the initial value.
    if not provided_state is None:
      initial_state = provided_state

    def cond(i, loop_inputs, loop_state, outputs):
      return i < size

    def loop(i, loop_inputs, loop_state, loop_outputs):
      inputs_i = map_fn(loop_inputs, [loop_inputs], lambda x: x[i])

      outputs_i, next_state, _ = self.forward_step(
        inputs_i,
        loop_state
      )

      Streamable.check_types_equality(
        Streamable.to_type(loop_state),
        Streamable.to_type(next_state),
        "Provided state and model state has different types (%s != %s)."
      )

      new_outputs = map_fn(outputs_i, [outputs_i, loop_outputs], lambda x, y: y.write(i, x))
      map_fn(loop_state, [loop_state, next_state], lambda x, y: tf.convert_to_tensor(y).set_shape(
        tf.convert_to_tensor(x).get_shape()))
      return (i + 1, loop_inputs, next_state, new_outputs)

    i0 = tf.constant(0)
    outputs = map_fn(outputs_0, [outputs_0],
      lambda x: tf.TensorArray(dtype=x.dtype, size=size))

    with tf.control_dependencies([assert_same_size]):
      i_f, inputs_f, state_f, outputs_f = tf.while_loop(cond, loop,
        loop_vars=[i0, inputs_tensors, initial_state, outputs], name="stream_loop")

    outputs = map_fn(outputs_f, [outputs_f], lambda o: o.stack())
    return (outputs, state_f, initial_state)

  def __call__(self, inputs, state=None, streamable=True):
    if streamable:
      return self.call_streamed(inputs, state)
    else:
      return self.forward_step(inputs, state) 

  def step(self, *inputs):
    """
    Where the internals of the operator must be implemented.
    """
    raise Exception("Not implemented")

  def forward_step(self, inputs, state):
    if not isinstance(inputs, (tuple, list)):
      inputs = (inputs,)
    if state is None:
      state = ()
    elif not isinstance(state, (tuple, list)):
      state = (state,)
    # By default we unpack inputs and state.
    return self.step(*inputs, *state)

  def concat(self, v1, v2):
    """
    Helper function concatenating two tensors together.
    This is used for concatenating shapes together.
    """
    if v1 == ():
      return v2
    elif v2 == ():
      return v1
    return tf.concat([v1, v2], axis=0)


class MetaStreamable(Streamable):
  def forward_step(self, inputs, state):
    # For a meta streamable, the operator does not know
    # the content of input and state as it can be anything.
    # Therefore, we simply pass the values or tuples to the
    # operator implementation. 
    return self.step(inputs, state)
