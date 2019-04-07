from setuptools import setup, find_packages

setup(
  name='tensorstream',
  version='0.0.4',
  author='Clement Michaud',
  author_email='clement.michaud34@gmail.com',
  license='Apache 2.0',

  packages=find_packages(exclude=['*_spec.py']),
  long_description=open('README.md').read(),
  install_requires=[
    "tensorflow==1.13.1",
    "pandas==0.24.2"
  ]
)
