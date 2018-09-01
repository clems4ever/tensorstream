
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
