"""
This file contains the main data structures of dataflow
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, QuantileTransformer, Binarizer
from sklearn.model_selection import train_test_split as train_test_split_sklearn

import pandas
import numpy

import inspect

class SmartDataFrame(pandas.DataFrame):
    def __init__(self, data=None, target=None, index=None, columns=None, dtype=None, copy=False):
        """SmartDataFrame Class wraps important scikit-learn data pre-processing methods over pandas.DataFrame

        The main idea behind SmartDataFrame is to allow the user to chain common data science logic on top of pandas DataFrame.

        Args:
            args: Arguments passed to a pandas.DataFrame object.
            kwargs: Keyword arguments passed to a pandas.DataFrame object.
        """
        super().__init__(data, index, columns, dtype, copy) 
        self.target = target

        # Preprocessing attributes
        self.label_encoder_registry = {}
        self.one_hot_encoder_registry = {}
        self.standard_scaler_registry = {}
        self.quantile_transformer_registry = {}
        self.binarizer_registry = {}

        self.label_encoded_features_keep = {}
        self.one_hot_encoded_features_keep = {}
        self.standard_scaled_features_keep = {}
        self.quantile_transformed_features_keep = {}
        self.binarized_features_keep = {}

        # Machine learning attributes
        self.model_registry = {}

    # Encoding methods
    def encode_labels(self, features, keep=False):
        """Encode labels with value between 0 and n_classes-1

        Args:
            features (list): The list of features to encode.
            keep (:obj:`bool`, optional): Whether or not the original features are kept in the processed dataset. Defaults to False.

        """
        def _encode_label_keep(self, feature):
            if not hasattr(self, "label_encoder_registry[feature]"):
                self.label_encoder_registry[feature] = LabelEncoder()
            self[feature + "_label_encoded"] = self.label_encoder_registry[feature].fit_transform(self[feature])

        def _encode_label(self, feature):
            if not hasattr(self, "label_encoder_registry[feature]"):
                self.label_encoder_registry[feature] = LabelEncoder()          
            self.label_encoder_registry[feature] = LabelEncoder()
            self[feature] = self.label_encoder_registry[feature].fit_transform(self[feature]) 

        if isinstance(features, (list, tuple)):
            for feature in features:
                self.label_encoded_features_keep[feature] = keep
                if keep:
                    _encode_label_keep(self, feature)
                else:
                    _encode_label(self, feature)
        elif isinstance(features, (str, int, float)):
            feature = features
            self.label_encoded_features_keep[feature] = keep
            if keep:
                _encode_label_keep(self, feature)
            else:
                _encode_label(self, feature)
        else:
            raise Exception

    def decode_labels(self, features=None, remove_registry_entry=False):
        if not features:
            features = list(self.label_encoded_features_keep.keys())

        for feature in features:
            if not self.label_encoded_features_keep[feature]:
                self[feature] = self.label_encoder_registry[feature].inverse_transform(self[feature])
                if remove_registry_entry:
                    del self.label_encoded_features_keep[feature]

    def one_hot_encode(self, parameters, keep=False):
        """Encode categorical integer features using a one-of-K scheme
        
        Args:
            parameters (dict): keys are DataFrame feature names and values are parameters for the sklearn OneHotEncoder"""

        pass

    # Scaling methods
    def standard_scaling(self, parameters_per_feature, keep=False):
        """ 
        parameters_per_feature (dict)"""
        pass

    # Simplifying methods
    def quantile_transform(self, parameters_per_feature, keep=False):
        """ 
        parameters_per_feature (dict)"""
        pass

    def binarize(self, parameters_per_feature, keep=False):
        """ 
        parameters_per_feature (dict)"""
        pass

    # Cleaning methods
    def remove_outlier(self, threshold):
        pass

    def fill_nan(self, **value_by_features):
        pass

    def delete_rows(self, callables):
        if isinstance(callables, callable):
            pass
        elif isinstance(callables, (list, tuple)):
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
                    raise ValueError("`target` must have 2 dimensions, not {}".format(len(target.shape)))
                if target.shape != [self.shape[0],1]:
                    raise ValueError("target shape must be {}, got {} instead".format(self.shape, target.shape))
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


class SmartArray(numpy.array):
    def __init__(self, data, target=None, dtype=None, copy=True, order='K', subok=False, ndmin=0):
        """SmartArray Class wraps important scikit-learn data pre-processing methods over numpy.array

        The main idea behind SmartArray is to allow the user to chain common data science logic on top of numpy array.

        Args:
            args: Arguments passed to a numpy array object.
            kwargs: Keyword arguments passed to a numpy array object.
        """
        super().__init__(data, dtype, copy, order, subok, ndmin) 


