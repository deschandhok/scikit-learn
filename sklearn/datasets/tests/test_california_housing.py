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
def test_fetch_california_housing_false():
    data = fetch_california_housing(download_if_missing=True)
    assert_equal(data.data.shape, (20640, 8))
    assert_equal(data.target.shape, (20640,))

    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]
    assert_array_equal(feature_names, data.feature_names)
   

#my tests above
