"""
This file contains the main data structures of dataflow
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, QuantileTransformer, Binarizer
from sklearn.model_selection import train_test_split as train_test_split_sklearn

import pandas
import numpy

import inspect

class SmartArray(numpy.array):
    def __init__(self, data, target=None, dtype=None, copy=True, order='K', subok=False, ndmin=0):
        """SmartArray Class wraps important scikit-learn data pre-processing methods over numpy.array

        The main idea behind SmartArray is to allow the user to chain common data science logic on top of numpy array.

        Args:
            args: Arguments passed to a numpy array object.
            kwargs: Keyword arguments passed to a numpy array object.
        """
        super().__init__(data, dtype, copy, order, subok, ndmin) 