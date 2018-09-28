import tensorflow as tf
from tensorstream.streamable import MetaStreamable

class Automaton(MetaStreamable):
  def __init__(self, transitions, init_state, default_action):
    self.transitions = transitions
    self.init_state = init_state
    self.default_action = default_action

  def initial_state(self, *inputs):
    return self.init_state

  def step(self, inputs, prev_state):
    cases = []
    for t in self.transitions:
      cond = tf.logical_and(
        tf.equal(prev_state, t[0]),
        t[2](*inputs)
      )
      output = t[3](*inputs)
      output_fn = lambda b_out=output, b_s=t[1]: (b_out, b_s)
      cases.append(( cond, output_fn ))

    def_value = self.default_action(*inputs)

    return tf.case(
      cases, default=lambda: (def_value, prev_state)
    )
