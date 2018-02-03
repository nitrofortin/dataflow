import pandas as pd
import numpy as np
import pytest

from dataflow.frame import SmartDataFrame


@pytest.fixture
def data():
    size = 100
    data = pd.DataFrame()
    data['label_1'] = ['a','b','c','d','e']*(size/5)
    data['label_2'] = ['f','g','h','i','j']*(size/5)
    data['numeric'] = np.arange(size)
    return SmartDataFrame(data=data)


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
            for funcargs in funcarglist])

class TestSuiteFrame(object):
    params = {
        'test_label_encode': [dict(features=i, keep=j) for i in ['label_1', \
                              ['label_1', 'label_2']] for j in [False,True]]

    }

    def test_label_encode(self, data, features, keep):
        data.label_encode(features=features, keep=keep)

    