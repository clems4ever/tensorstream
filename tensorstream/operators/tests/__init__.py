import os
import unittest
import pandas as pd
from pyexcel_ods import get_data

class TestCase(unittest.TestCase):
  def read_ods(self, filename, header=0):
    data = get_data(filename)
    return {k: pd.DataFrame(d[header+1:], columns=d[header])
      .set_index('Date') for k, d in data.items()}

  def r_ods(self, filename):
    return get_data(filename)

  def read_csv(self, filename):
    return pd.read_csv(filename, sep=',', 
      header=0, parse_dates=['Date'], index_col='Date')

  def from_test_res(self, relative_path, test_file_path=None):
    if test_file_path is None:
      dir_path = os.path.dirname(os.path.realpath(__file__))
    else:
      dir_path = os.path.join(os.path.dirname(os.path.realpath(test_file_path)), 'tests')
    return os.path.join(dir_path, 'resources', relative_path)
