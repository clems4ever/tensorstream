from tensorstream.operators import Streamable

class Compose(Streamable):
  def __init__(self, *operators):
    super().__init__(operators[0].dtype, operators[0].shape)

    self.operators = tuple(reversed(operators))
    self.initial_state = tuple(map(lambda x: x.initial_state, self.operators))

  def step(self, *inputs_and_states):
    next_states = []

    istate = self.initial_state

    total_len = len(inputs_and_states)
    state_len = len(self.initial_state)
    out_len = len(self.dtype) if isinstance(self.dtype, tuple) else 1
    in_len = total_len - state_len

    states = inputs_and_states[in_len:] if isinstance(istate, tuple) else inputs_and_states[-1]
    inputs = inputs_and_states[0:in_len]

    op_inputs = inputs
    op_states = list(zip(self.operators, states))

    for operator, state in op_states:
      output, next_state = operator(*op_inputs, state=state)
      op_inputs = output if isinstance(output, tuple) else (output,)
      next_states.append(next_state)

    return output, tuple(next_states)
