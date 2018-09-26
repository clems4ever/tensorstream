from setuptools import setup, find_packages

setup(
  name='tensorstream',
  version='0.0.1',
  author='Clement Michaud',
  author_email='clement.michaud34@gmail.com',
  license='All rights reserved to Clement Michaud',

  packages=find_packages(exclude=['*_spec.py']),
  long_description=open('README').read(),
  install_requires=[
    "tensorflow==1.5.0",
    "pandas==0.23.4"
  ],
  extras_require={
    "dev": [
      "pytest",
      "pyexcel-ods==0.5.3"
    ]
  }
)
