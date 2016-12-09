import errno
import scipy.sparse as sp
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets.species_distributions import construct_grids
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true, assert_raises
from sklearn.utils.testing import SkipTest

#my tests below

def test_fetch_california_housing_true():
    assert_raises(IOError, fetch_california_housing, download_if_missing=False)
    fetch_california_housing(download_if_missing=True)
    fetch_california_housing(download_if_missing=False)

#my tests above
