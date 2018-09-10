from tensorstream.streamable import MetaStreamable

class Compose(MetaStreamable):
  def __init__(self, *operators):
    super().__init__()
    self.operators = tuple(reversed(operators))

  def initial_state(self, *inputs):
    x = inputs
    states = []
    for op in self.operators:
      y, _ = op(x, streamable=False)
      if isinstance(x, (tuple, list)):
        states.append(op.initial_state(*x))
      else:
        states.append(op.initial_state(x))
      x = y
    return tuple(states)

  def step(self, inputs, states):
    next_states = []
    op_inputs = inputs
    op_states = list(zip(self.operators, states))

    for operator, state in op_states:
      output, next_state = operator(inputs=op_inputs, state=state, streamable=False)
      op_inputs = output
      next_states.append(next_state)

    return output, tuple(next_states)
