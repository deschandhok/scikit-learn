"""Test the covtype loader.

Skipped if covtype is not already downloaded to data_home.
"""

from sklearn.datasets import fetch_covtype
from sklearn.utils.testing import assert_equal, assert_array_equal, SkipTest, assert_false
import numpy as np
#my tests below

def test_fetch_covtype_true_shuffle():
    hold1 = fetch_covtype(download_if_missing=True, shuffle = True)
    hold2 = fetch_covtype(download_if_missing=False, shuffle = False)

    data1, data2 = hold1['data'], hold2['data']
    target1, target2 = hold1['data'], hold2['data']

    assert_false(np.array_equal(data1, data2))
    assert_false(np.array_equal(target1, target2))

#my tests above


def fetch(*args, **kwargs):
    return fetch_covtype(*args, download_if_missing=False, **kwargs)


def test_fetch():
    try:
        data1 = fetch(shuffle=True, random_state=42)
    except IOError:
        raise SkipTest("Covertype dataset can not be loaded.")

    data2 = fetch(shuffle=True, random_state=37)

    X1, X2 = data1['data'], data2['data']
    assert_equal((581012, 54), X1.shape)
    assert_equal(X1.shape, X2.shape)

    assert_equal(X1.sum(), X2.sum())

    y1, y2 = data1['target'], data2['target']
    assert_equal((X1.shape[0],), y1.shape)
    assert_equal((X1.shape[0],), y2.shape)
