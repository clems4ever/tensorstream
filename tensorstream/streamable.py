import tensorflow as tf
import functools

from tensorstream.helpers.map_fn import map_fn
from tensorstream.helpers.flatten import flatten

class Streamable:
  def __init__(self, initial_state=()):
    # If there is no method called "initial_state", we set the property
    # from parameters (for backward compatibility)..
    if not ("initial_state" in dir(self) and callable(
      getattr(self, "initial_state"))):
      if callable(initial_state):
        self.initial_state = initial_state
      else:
        self.initial_state = lambda *kargs: initial_state

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

  def _get_operator_properties(self, initial_state, inputs):
    operator_initial_state = self._deduce_initial_state(inputs)
    if initial_state is None:
      initial_state = operator_initial_state
    else:
      # Check provided state type is compatible with operator initial state type.
      Streamable.check_types_equality(
        Streamable.to_type(operator_initial_state),
        Streamable.to_type(initial_state),
        "Input state has wrong type. operator.initial_state: %s, input_state: %s."
      )

    outputs, _ = self.forward_step(inputs, initial_state)
    dtype = map_fn(outputs, [outputs], lambda o: tf.convert_to_tensor(o).dtype)
    shape = map_fn(outputs, [outputs], lambda o: tf.shape(o))
    return dtype, shape, initial_state


  def call_streamed(self, inputs_tensors, initial_state):
    inputs_sizes = tuple(map(lambda x: tf.shape(x)[0], flatten(inputs_tensors)))
    size = inputs_sizes[0]
    same_size = functools.reduce(lambda acc, x: tf.logical_and(acc, tf.equal(x, size)),
      inputs_sizes, tf.constant(True))

    assert_same_size = tf.assert_equal(same_size, tf.constant(True),
      data=inputs_sizes, message="inputs have different sizes")

    first_inputs = map_fn(inputs_tensors, [inputs_tensors], lambda x: x[0])
    dtype, shape, initial_state = self._get_operator_properties(initial_state, first_inputs)

    def cond(i, loop_inputs, loop_state, outputs):
      return i < size

    def loop(i, loop_inputs, loop_state, loop_outputs):
      inputs_i = map_fn(loop_inputs, [loop_inputs], lambda x: x[i])

      current_inputs = inputs_i
      previous_state = loop_state

      outputs_i, next_state = self.forward_step(
        inputs_i,
        loop_state
      )

      Streamable.check_types_equality(
        Streamable.to_type(initial_state),
        Streamable.to_type(next_state),
        "Output state has wrong type. operator.initial_state: %s, output_state: %s."
      )

      new_outputs = map_fn(outputs_i, [outputs_i, loop_outputs], lambda x, y: y.write(i, x))
      map_fn(loop_state, [loop_state, next_state], lambda x, y: y.set_shape(x.get_shape()))
      return (i + 1, loop_inputs, next_state, new_outputs)

    i0 = tf.constant(0)

    outputs = map_fn(dtype, [dtype],
      lambda x: tf.TensorArray(dtype=x, size=size))

    with tf.control_dependencies([assert_same_size]):
      i_f, inputs_f, state_f, outputs_f = tf.while_loop(cond, loop,
        loop_vars=[i0, inputs_tensors, initial_state, outputs], name="stream_loop")
    outputs = map_fn(outputs_f, [outputs_f], lambda o: o.stack())

    return (outputs, state_f)

  def _deduce_initial_state(self, inputs):
    if "initial_state" in dir(self) and callable(getattr(self, "initial_state")):
      if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
      state = self.initial_state(*inputs)
    else:
      state = self.initial_state
    return state

  def __call__(self, inputs, state=None, streamable=True):
    if streamable:
      return self.call_streamed(inputs, state)
    else:
      # If no state provided, we deduce it.
      if state is None:
        state = self._deduce_initial_state(inputs)
      return self.forward_step(inputs, state) 


  def step(self, *inputs):
    """
    Where the internals of the operator must be implemented.
    """
    raise Exception("Not implemented")

  def forward_step(self, inputs, state):
    if not isinstance(inputs, (tuple, list)):
      inputs = (inputs,)
    if not isinstance(state, (tuple, list)):
      state= (state,)
    # By default we unpack inputs and state.
    return self.step(*inputs, *state)

  def concat(self, v1, v2):
    """
    Helper function concatenating two tensors together.
    This is used for concatenating shapes together.
    """
    return tf.concat([v1, v2], axis=0)


class MetaStreamable(Streamable):
  def __init__(self, initial_state=()):
    super().__init__(initial_state)

  def forward_step(self, inputs, state):
    # For a meta streamable, the operator does not know
    # the content of input and state as it can be anything.
    # Therefore, we simply pass the values or tuples to the
    # operator implementation. 
    return self.step(inputs, state)
