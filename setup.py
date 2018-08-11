from setuptools import setup

setup(
  name='TensorStream',
  version='0.1',
  packages=['tensorstream',],
  license='All rights reserved to Clement Michaud',
  long_description=open('README').read(),
  install_requires=[
    "tensorflow == 1.5.0",
    "pandas==0.23.4"
  ],
  extras_require={
    'dev': [
      'pytest',
      'pyexcel-ods==0.5.3'
    ]
  }
)
