import pandas as pd
import numpy as np
import pytest

from dataflow.frame import SmartDataFrame


@pytest.fixture
def data():
	size = 100
	data = pd.DataFrame()
	data['label'] = ['a','b','c','d','e']*(size/5)
	data['numeric'] = np.arange(size)
	return SmartDataFrame(data=data)


def test_label_encode(data):
	data.label_encode(features='label')