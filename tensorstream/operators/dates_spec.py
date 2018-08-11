import unittest
import tensorflow as tf
import numpy as np

from tensorstream.operators import dates

class DatesSpec(unittest.TestCase):
  def setUp(self):
    self.session = tf.Session()

  def assertTensor(self, tensor):
    assert(tensor.eval(session=self.session))
  def assertTensorNot(self, tensor):
    assert(not tensor.eval(session=self.session))
    
  def test_less_equal(self):
    date1 = tf.constant([1, 1, 2016])
    date2 = tf.constant([8, 1, 2016])
    date3 = tf.constant([1, 4, 2016])
    date4 = tf.constant([1, 1, 2018])
    date5 = tf.constant([20, 8, 2015])
    date6 = tf.constant([20, 8, 2018])

    # equal dates
    self.assertTensor(dates.less_equal(date1, date1))

    # diff in days
    self.assertTensor(dates.less_equal(date1, date2))
    self.assertTensorNot(dates.less_equal(date2, date1))

    # diff in months
    self.assertTensor(dates.less_equal(date1, date3))
    self.assertTensorNot(dates.less_equal(date3, date1))

    # diff in years
    self.assertTensor(dates.less_equal(date1, date4))
    self.assertTensorNot(dates.less_equal(date4, date1))
    
    # multiple diffs
    self.assertTensor(dates.less_equal(date5, date1))
    self.assertTensor(dates.less_equal(date1, date6))

  def test_is_between(self):
    date1 = [1, 1, 2016]
    date2 = [8, 1, 2016]
    date3 = [1, 4, 2017]
    date4 = [6, 9, 2015]
  
    self.assertTensor(dates.between(date2, date1, date3))
    self.assertTensorNot(dates.between(date4, date1, date3))
    self.assertTensor(dates.between(date2, date2, date2))
   
  def test_select_in_range(self):
    dates_ts = tf.constant([
      [1, 1, 2016],
      [2, 1, 2016],
      [3, 1, 2016],
      [6, 1, 2016],
      [9, 1, 2016],
      [1, 2, 2016],
      [5, 2, 2016],
      [9, 3, 2016],
      [1, 1, 2017],
      [1, 3, 2017],
    ])
    def test(start_date, end_date, expected_output):
      selected_dates = self.session.run(
        dates.in_range(start_date, end_date, dates_ts))
      expected = np.array(expected_output, dtype=np.bool)
      np.testing.assert_equal(selected_dates, expected)

    test(tf.constant([7, 1, 2016]), tf.constant([1, 12, 2016]), [
      False, False, False, False, True, True, True, True, False, False])
    test(tf.constant([1, 1, 2015]), tf.constant([1, 12, 2018]), [
      True, True, True, True, True, True, True, True, True, True])
    test(tf.constant([1, 1, 2018]), tf.constant([1, 12, 2019]), [
      False, False, False, False, False, False, False, False, False, False])
    test(tf.constant([5, 2, 2016]), tf.constant([6, 2, 2016]), [
      False, False, False, False, False, False, True, False, False, False])
