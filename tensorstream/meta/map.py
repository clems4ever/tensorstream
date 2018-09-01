import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream import map_fn

class Map(Streamable):
  def __init__(self, operator, size):
    shape = map_fn(operator.dtype, [operator.shape], lambda s: (size,) + s)
    super().__init__(operator.dtype, shape)

    self.size = size
    self.operator = operator
    self.initial_state = map_fn(operator.initial_state, [operator.initial_state],
      lambda x: tf.tile(tf.expand_dims(x, axis=0), [size] + [1] * x.get_shape().ndims))
 
  def step(self, *inputs_and_states):
    total_len = len(inputs_and_states)
    state_len = len(self.initial_state)
    in_len = total_len - state_len

    inputs = inputs_and_states[:in_len]
    previous_states = inputs_and_states[in_len:]

    state_dtype = map_fn(previous_states, [previous_states], lambda x: x.dtype) 
    output_dtype = self.operator.dtype
    def apply_op(inputs_states):
      return self.operator(inputs_states[0], state=inputs_states[1], streamable=False)
    outputs, next_states = tf.map_fn(apply_op, (inputs, previous_states), dtype=(output_dtype, state_dtype))
    return outputs, next_states
