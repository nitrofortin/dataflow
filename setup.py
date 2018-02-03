from distutils.core import setup

from setuptools import setup, find_packages

setup(name='dataflow',
      version='0.1',
      description='wrapper of numpy, pandas and sklearn',
      classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='wrapper pandas sklearn numpy',
      url='https://github.com/nitrofortin/dataflow',
      author='Flying Circus',
      packages=find_packages(),
      install_requires=[
          'numpy','pandas','python-dateutil',
          'pytz', 'scikit-learn', 'six', 'sklearn'
      ],
      include_package_data=True,
      zip_safe=False)
