"""
This file contains the main data structure of dataflow
"""

from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler, 
                                   QuantileTransformer, Binarizer)
from sklearn.model_selection import train_test_split as train_test_split_sklearn

import pandas
import numpy
import copy 
import functools
import collections 

import inspect
from builtins import *

from .series import SmartSeries

sklearn_prep_map = {
    'label_encode': LabelEncoder,
    'one_hot_encode': OneHotEncoder,
    'standard_scale': StandardScaler,
    'quantile_transform': QuantileTransformer,
    'binarize': Binarizer
}

sklearn_pre_undo_map = {
    'label_decode': 'label_encode',
    'one_hot_decode': 'one_hot_encode',
    'standard_unscale': 'standard_scale',
    'quantile_untransform': 'quantile_transform',
    'unbinarize': 'binarize'
}

class SmartDataFrame(pandas.DataFrame):
    def __init__(self, data=None, target=None, index=None, columns=None, 
                 dtype=None, copy=False):
        """
        SmartDataFrame Class wraps important scikit-learn data pre-processing 
        methods over pandas.DataFrame

        The main idea behind SmartDataFrame is to allow the user to chain common 
        data science logic on top of pandas DataFrame.

        Args:
            args: Arguments passed to a pandas.DataFrame object.
            kwargs: Keyword arguments passed to a pandas.DataFrame object.
        """

        super().__init__(data, index, columns, dtype, copy) 
        self.target = target

        # Preprocessing attributes
        self.registry = collections.defaultdict()
        self.features_keep = collections.defaultdict()

        # Machine learning attributes
        self.model_registry = {}

    @property
    def _constructor(self):
        return SmartDataFrame

    @property
    def _constructor_sliced(self):
        return SmartSeries

    def _copy(self, deep=True):
        results = self
        if deep:
            # results = self.copy()
            results = self._constructor(copy.deepcopy(self._data))
            for k, v in self.__dict__.items():
                if k != '_data':
                    results.__dict__[k] = v
        return results

    def sklearn_preprocessing(f):
        @functools.wraps(f)
        def __method(self, features, keep_original=True, inplace=False, 
                         **sklearn_kwargs):
            method_name = f.__name__
            result = self if inplace else self._copy()

            if method_name not in result.features_keep:
                result.features_keep[method_name] = {}
                result.registry[method_name] = {}

            if not isinstance(features, (list, tuple)):
                features = [features]

            for feature in features:
                code_name = "{}_{}".format(feature, method_name) 
                result.features_keep[method_name][feature] = keep_original
                result.registry[method_name][feature] = sklearn_prep_map[method_name](**sklearn_kwargs)

                result.__setitem__(code_name, result.registry[method_name][feature] \
                                        .fit_transform(result[feature]))
                if not keep_original:
                    del result[feature]
            if not inplace:
                return result
        return __method

    def sklearn_preprocessing_undo(f):
        @functools.wraps(f)
        def __method(self, features, keep_original=True, inplace=False, 
                         **sklearn_kwargs):
            method_name = sklearn_pre_undo_map[f.__name__]
            result = self if inplace else self._copy()
            
            if not features:
                features = list(self.features_keep[method_name].keys())

            if not isinstance(features, (list, tuple)):
                features = [features]

            for feature in features:
                code_name = feature.split('_label_encode')[0]
                if code_name in result.registry[method_name].keys():

                    result.__setitem__(code_name, result \
                            .registry[method_name][code_name] \
                            .inverse_transform(result[feature]))
                    if not keep_original:
                        del result[feature]
                    
                else:
                    raise Exception('Cannot decode feature {}, try among {}' 
                            .format(feature, self.registry[method_name][code_name].keys()))
            return result
        return __method

    # Encoding methods
    @sklearn_preprocessing
    def label_encode(self, features, keep_original=True, inplace=False, 
                     **sklearn_kwargs):
        """Encode labels with value between 0 and n_classes-1

        Args:
            features (list): The list of features to encode.
            keep (:obj:`bool`, optional): Whether or not the original features 
                are kept in the processed dataset. Defaults to False.
            sklearn_kwargs (dict): other parameters to pass to scikit-learn 
                LabelEncoder.
        """
        pass

    @sklearn_preprocessing_undo
    def label_decode(self, features=None, keep_original=True, inplace=False):
        pass

    @sklearn_preprocessing
    def one_hot_encode(self, features, keep_original=True, inplace=False, 
                     **sklearn_kwargs):
        """Encode categorical features using a one-of-K scheme
        
        Args:
            features: features to one-hot encode.
            keep (:obj:`bool`, optional): Whether or not the original features 
                are kept in the processed dataset. Defaults to False.
            sklearn_kwargs (dict): other parameters to pass to scikit-learn 
                OneHotEncoder.
        """
        pass
            
    @sklearn_preprocessing_undo
    def one_hot_decode(self, features=None, keep_original=True, inplace=False):
        pass

    # Scaling methods
    @sklearn_preprocessing
    def standard_scale(self, features, keep_original=True, inplace=False, 
                     **sklearn_kwargs):
        """Standardize features by removing the mean and scaling to unit variance
        
        Args:
            features: features to scale.
            keep (:obj:`bool`, optional): Whether or not the original features 
                are kept in the processed dataset. Defaults to False.
            sklearn_kwargs (dict): other parameters to pass to scikit-learn 
                StandardScaler.
        """
        pass

    @sklearn_preprocessing_undo
    def standard_unscale(self, features=None, keep_original=True, inplace=False):
        pass
        
    # Simplifying methods
    @sklearn_preprocessing
    def quantile_transform(self, features, keep_original=True, inplace=False, 
                     **sklearn_kwargs):
        """Transform features using quantiles information
        
        Args:
            features: features to binarize.
            keep (:obj:`bool`, optional): Whether or not the original features 
                are kept in the processed dataset. Defaults to False.
            sklearn_kwargs (dict): other parameters to pass to scikit-learn 
                QuantileTransformer.
        """
        pass

    @sklearn_preprocessing_undo
    def quantile_untransform(self, features=None, keep_original=True, inplace=False):
        pass

    @sklearn_preprocessing
    def binarize(self, features, keep_original=True, inplace=False, 
                     **sklearn_kwargs):
        """Binarize data (set feature values to 0 or 1) according to a threshold
        
        Args:
            features: features to binarize.
            keep (:obj:`bool`, optional): Whether or not the original features 
                are kept in the processed dataset. Defaults to False.
            sklearn_kwargs (dict): other parameters to pass to scikit-learn 
                Binarizer.
        """
        pass

    @sklearn_preprocessing_undo
    def unbinarize(self, features=None, keep_original=True, inplace=False):
        pass

    # Cleaning methods
    def remove_outlier(self, threshold):
        pass

    def type_checking(self, types, rules):
        # given feature types and what to do if type condition not met (rules)
        pass

    def drop_features(self, features, inplace=False):
        """
        Args:
            inplace (bool): if True, drop features inplace and return self. 
        """

        if inplace:
            self = self.drop(features, axis=1)
            return self
        else:
            return self.drop(features, axis=1)

    # Data Mining
    def get_correlations(self, features):
        pass

    # Machine learning methods for quick prototyping
    def train_model(self, model, target=None):
        """
        Args:
            model (scikit-learn model object): (e.g. sklearn.tree.DecisionTreeRegressor())
            target ((:obj:`str`, :obj:`numpy.array`, :obj:`pandas.DataFrame), optional):  
        """

        if target:
            if isinstance(target, str):
                if target in self.columns:
                    target = self[target]
                else:
                    raise ValueError("`target` is not among SmartDataFrame columns")
            else:
                if not hasattr(target, "shape"):
                    raise ValueError("`shape` attribute not defined for given `target` object.")
                if len(target.shape) != 2:
                    raise ValueError("`target` must have 2 dimensions, not {}" \
                                     .format(len(target.shape)))
                if target.shape != [self.shape[0],1]:
                    raise ValueError("target shape must be {}, got {} instead" \
                                     .format(self.shape, target.shape))
        else:
            target = self.target

        if not hasattr(model, "fit"):
            raise Exception("`fit` method not defined for given `model` object.")

        model_hash = str(model).__hash__()
        self.model_registry[model_hash] = {}
        self.model_registry[model_hash]['model'] = model
        self.model_registry[model_hash]['hyper_parameters'] = model.__dict__
        self.model_registry[model_hash]["trained"] = False

        model_fit_signature = inspect.signature(model.fit)
        number_of_mandatory_arguments = 0

        for parameter in model_fit_signature.parameters.values():
            # If the parameter is mandatory
            if parameter.default is inspect._empty:
                number_of_mandatory_arguments += 1

        # If the model is supervised
        if number_of_mandatory_arguments is 2:
            X = self.ix[:, self.columns != target]
            y = target
            self.model_registry[model_hash]['model'].fit(X, y)

        # If the model is unsupervised
        elif number_of_mandatory_arguments is 1:
            self.model_registry[model_hash]['model'].fit(self)
        
        else:
            raise Exception("Unknown number of mandatory arguments")

        self.model_registry[model_hash]["trained"] = True

        return self.model_registry[model_hash]['model']

    def get_model(self, model):
        model_hash = str(model).__hash__()
        try:
            return self.model_registry[model_hash]
        except:
            raise Exception("Model not found")

    def delete_model(self, model):
        model_hash = str(model).__hash__()
        try:
            del self.model_registry[model_hash]
        except:
            raise Exception("Model not found")

    # Model selection methods
    def train_test_split(self, **options):
        return train_test_split_sklearn(self, **options)

    def evaluate_model(self, model, evaluation_method):
        pass

    # def predict(self, model, data)
    #     try:
    #         return model.predict(data)
    #     except:   
    #         raise Exception("Model must be trained")
        
    # def predict_log_proba(self, model, data)
    #     try:
    #         return model.predict_log_proba(data)
    #     except:   
    #         raise Exception("Model must be trained")

    # def predict_proba(self, model, data)
    #     try:
    #         return model.predict_proba(data)
    #     except:   
    #         raise Exception("Model must be trained")




