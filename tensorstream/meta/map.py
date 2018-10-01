import tensorflow as tf

from tensorstream.helpers.map_fn import map_fn
from tensorstream.streamable import MetaStreamable

class Map(MetaStreamable):
  def __init__(self, operator, size):
    super().__init__()
    self.size = size
    self.operator = operator

  def step(self, inputs, states=None):
    def dimensions(x):
      dim = tf.ones([tf.size(tf.shape(x))], dtype=tf.int32)
      return tf.concat([[self.size], dim], axis=0)

    # Run the model without state to get the dtype
    op_value, next_state, init_state = self.operator(
      inputs[0], streamable=False)

    if states is None:
      extended_init_state = map_fn(init_state, [init_state],
        lambda x: tf.tile(tf.expand_dims(x, axis=0), dimensions(x)))
      states = (extended_init_state,)

    prev_state = states[0]

    state_dtype = map_fn(prev_state, [prev_state], lambda x: x.dtype) 
    output_dtype = map_fn(op_value, [op_value], lambda o: o.dtype)

    def apply_op(inputs_states):
      outputs, new_state, _ = self.operator(
        inputs=inputs_states[0],
        state=inputs_states[1],
        streamable=False
      )
      return outputs, new_state

    outputs, next_state = tf.map_fn(
      apply_op,
      (inputs, prev_state),
      dtype=(output_dtype, state_dtype)
    )
    return outputs, (next_state,), (prev_state,)
