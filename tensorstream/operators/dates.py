import tensorflow as tf

def less_equal(date1, date2):
  d1, m1, y1 = tf.unstack(date1, 3)
  d2, m2, y2 = tf.unstack(date2, 3)
  return tf.logical_or(
    tf.less(y1, y2),
    tf.logical_and(
      tf.equal(y1, y2),
      tf.logical_or(
        tf.less(m1, m2),
        tf.logical_and(
          tf.equal(m1, m2),
          tf.less_equal(d1, d2)
        )
      )
    )
  )

def between(date, sdate, edate):
  return tf.logical_and(
    less_equal(sdate, date),
    less_equal(date, edate),
  )

def in_range(start_date, end_date, dates):
  return tf.map_fn(lambda date: between(date, start_date, end_date),
    dates, dtype=tf.bool)
