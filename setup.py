from distutils.core import setup

setup(name='datapho',
      version='1.0',
      description='Python Data Science Library Wrapper',
      author='Julien St-Pierre Fortin',
      packages=['datapho'],
      install_requires=['pandas>=0.21.0','numpy>=1.13.1', 'matplotlib>=2.1.0'],
     )