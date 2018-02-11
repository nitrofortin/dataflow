import pandas as pd
import numpy as np
import pytest
import itertools

from dataflow.frame import SmartDataFrame


@pytest.fixture
def data():
    size = 100
    data = pd.DataFrame()
    data['label_1'] = ['a','b','c','d','e']*int(size/5)
    data['label_2'] = ['f','g','h','i','j']*int(size/5)
    data['numeric'] = np.arange(size)
    return SmartDataFrame(data=data)


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
            for funcargs in funcarglist])

class TestUnitFrame(object):
    """Test suite for SmartDataFrame methods
    TODO: test remaining preprocessing methods
    """
    params = {
        'test_encode_label': [dict(features=i[0], keep=i[1], inplace=i[2]) for i \
                in itertools.product(['label_1',['label_1', 'label_2']], \
                [True,False], [True, False])],

        'test_decode_label': [dict(features=i, keep=j) for i in \
                              ['label_1_label_encode', None] for j in \
                              [False, True]]
    }

    def test_encode_label(self, data, features, keep, inplace):
        result = data.label_encode(features=features, keep_original=keep, 
                                   inplace=inplace)
        if inplace:
            result = data

        if isinstance(features, list):
            assert all("{}_label_encode".format(feature) in result.columns 
                       for feature in features)
        else:
            assert "{}_label_encode".format(features) in result.columns

        if keep:
            if isinstance(features, list):
                assert all(feature in result.columns for feature in features)
            else:
                assert features in result.columns            

    def test_decode_label(self, data, features, keep):
        res = data.label_encode(['label_1','label_2'], keep_original=False)
        assert 'label_1_label_encode' in res.columns
        assert 'label_2_label_encode' in res.columns   
        assert 'label_1_label_encode' not in data.columns
        assert 'label_2_label_encode' not in data.columns      
        res.label_decode(features=['label_1_label_encode','label_2_label_encode'],
                         keep_original=False, inplace=True)
        assert 'label_1_label_encode' not in res.columns
        assert 'label_2_label_encode' not in res.columns

