import errno
import scipy.sparse as sp
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal, assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import SkipTest

#stuff I did is below
def test_species_distributions_true():
    data = fetch_olivetti_faces(shuffle= True, download_if_missing=True)

    assert_equal(data.data.shape, (400, 4096))
    assert_equal(data.images.shape, (400, 64, 64))
    assert_equal(data.target.shape, (400, ))


#stuff I did is above
