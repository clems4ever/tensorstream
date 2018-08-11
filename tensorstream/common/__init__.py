import pandas as pd
import tensorflow as tf

def map_consecutive_fn(fn, elems):
  cons_couples = tf.transpose(tf.stack([elems[1:], elems[:-1]]))
  return tf.map_fn(lambda x: fn(x[0], x[1]), cons_couples, dtype=tf.bool)

def filter_with_mask(mask, values):
  size = tf.size(mask)
  def cond(i, output_index, output):
    return i < size 
  def loop(i, output_index, output):
    new_output = tf.cond(mask[i],
      lambda: (output_index + 1, output.write(output_index, values[i])),
      lambda: (output_index, output))
    return (i+1, *new_output)
  i0 = tf.constant(0)
  output_index = tf.constant(0)
  output = tf.TensorArray(values.dtype, size=tf.reduce_sum(tf.to_int32(mask)))
  results = tf.while_loop(cond, loop, loop_vars=[i0, output_index, output])
  return results[2].stack()

def shift(new_value, vector, axis=0):
  return tf.concat([[new_value], 
        tf.slice(vector, [0], [tf.size(vector) - 1])], axis)

def roll(value, tensor, shift=1, axis=0):
  return tf.stack([value, *tf.unstack(tensor[:-1])])

def transpose(products_df):
  columns = list(products_df[0])
  data = {}
  for col in columns:
    series = list(map(lambda ts: ts[col], products_df))
    data[col] = pd.concat(series, axis=1)
  return data

