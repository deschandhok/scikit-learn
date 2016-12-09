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
    assert_raises(IOError, fetch_olivetti_faces, download_if_missing=False)
    fetch_olivetti_faces(shuffle= True, download_if_missing=True)
    fetch_olivetti_faces(shuffle= False, download_if_missing=False)

#stuff I did is above
