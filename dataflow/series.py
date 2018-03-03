import pandas
import numpy

from .frame import SmartDataFrame

class SmartSeries(pandas.Series):
    def __init__(self, data=None, index=None, columns=None, 
                 dtype=None, copy=False):
        """
        SmartSeries Class wraps important scikit-learn data pre-processing 
        methods over pandas.Series

        The main idea behind SmartSeries is to allow the user to chain common 
        data science logic on top of pandas Series.

        Args:
            args: Arguments passed to a pandas.Series object.
            kwargs: Keyword arguments passed to a pandas.Seires object.
        """

        super().__init__(data, index, columns, dtype, copy) 

        # Preprocessing attributes
        self.registry = collections.defaultdict()

    @property
    def _constructor(self):
        return SmartSeries

    @property
    def _constructor_expanddim(self):
        return SmartDataFrame