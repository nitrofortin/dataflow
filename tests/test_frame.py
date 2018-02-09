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
    params = {
        'test_encode_label': [dict(features=i[0], keep=i[1], inplace=i[2]) for i \
                in itertools.product(['label_1',['label_1', 'label_2']], \
                [True,False], [True, False])],

        'test_decode_label': [dict(features=i, keep=j) for i in \
                              ['label_1_label_encoded', None] for j in \
                              [False, True]]
    }

    def test_encode_label(self, data, features, keep, inplace):
        data.encode_label(features=features, keep_original=keep, inplace=inplace)

    def test_decode_label(self, data, features, keep):
        data.encode_label(features='label_1', inplace=True)
        data.decode_label(features=features, keep_original=keep)

class TestFunctionalFrame(object):
    pass
