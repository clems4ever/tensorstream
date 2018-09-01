
def map_fn(obj, objs, apply_fn):
  if isinstance(obj, dict):
    return dict(map(lambda kv: (kv[0], map_fn(kv[1], map(lambda x: x[kv[0]], objs), apply_fn)), obj.items()))
  elif isinstance(obj, list):
    return list(map(lambda e: map_fn(e[0], e[1:], apply_fn), zip(obj, *objs)))
  elif isinstance(obj, tuple):
    return tuple(map(lambda e: map_fn(e[0], e[1:], apply_fn), zip(obj, *objs)))
  else:
    return apply_fn(*objs)
