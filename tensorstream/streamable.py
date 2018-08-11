import tensorflow as tf
import functools

def traverse(obj1, objs, apply_fn):
  if isinstance(obj1, dict):
    def fn(kv):
      new_objs = list(map(lambda o: o[kv[0]], objs))
      return (kv[0], traverse(kv[1], new_objs, apply_fn))
    return dict(map(fn, obj1.items()))
  elif isinstance(obj1, list):
    return list(map(lambda e: traverse(e[0], e[1:], apply_fn), zip(obj1, *objs)))
  elif isinstance(obj1, tuple):
    return tuple(map(lambda e: traverse(e[0], e[1:], apply_fn), zip(obj1, *objs)))
  else:
    return apply_fn(*objs)

def flatten(elems):
  stack = []
  if isinstance(elems, (list, tuple)):
    for x in elems:
      stack += flatten(x)
  elif isinstance(elems, (dict)):
    for x in elems.values():
      stack += flatten(x)
  else:
    stack.append(elems)
  return stack

class Streamable:
  def __init__(self, dtype=tf.int32, shape=(), initial_state=()):
    self.dtype = dtype
    self.shape = shape
    self.initial_state = initial_state

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
      inputs_i = traverse(loop_inputs, [loop_inputs], lambda x: x[i])
      if isinstance(loop_inputs, (tuple, list)):
        current_inputs = inputs_i
      else:
        current_inputs = (inputs_i,)

      if isinstance(loop_state, (tuple, list)):
        previous_state = loop_state
      else:
        previous_state= (loop_state,)
        
      outputs_i, next_state = self.step(*current_inputs, *previous_state)

      new_outputs = traverse(outputs_i, [outputs_i, loop_outputs], lambda x, y: y.write(i, x))
      traverse(loop_state, [loop_state, next_state], lambda x, y: y.set_shape(x.get_shape()))
      return (i + 1, loop_inputs, next_state, new_outputs)

    i0 = tf.constant(0)

    outputs = traverse(self.dtype, [self.dtype, self.shape],
      lambda x, y: tf.TensorArray(dtype=x, element_shape=y, size=size))

    with tf.control_dependencies([assert_same_size]):
      i_f, inputs_f, state_f, outputs_f = tf.while_loop(cond, loop,
        loop_vars=[i0, inputs_tensors, state, outputs], name="stream_loop")
    outputs = traverse(outputs_f, [outputs_f], lambda o: o.stack())

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

