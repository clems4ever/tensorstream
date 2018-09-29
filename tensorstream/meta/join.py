from tensorstream.streamable import MetaStreamable

class Join(MetaStreamable):
  def __init__(self, *operators):
    super().__init__()
    self.operators = operators

  def properties(self, *inputs):
    initial_states = []
    placeholders = []
    op_in = zip(self.operators, inputs)
    for op, inp in op_in:
      if isinstance(inp, (list, tuple)):
        y, init_state = op.properties(*inp)
      else:
        y, init_state = op.properties(inp)
      initial_states.append(init_state)
      placeholders.append(y)
    return tuple(placeholders), tuple(initial_states)

  def step(self, inputs, states):
    outputs = []
    next_states = []
    op_in_st = zip(self.operators, inputs, states)
    for op, inputs_, state in op_in_st:
      output, next_state = op(
        inputs_, state, streamable=False)
      outputs.append(output)
      next_states.append(next_state)
    return tuple(outputs), tuple(next_states)
