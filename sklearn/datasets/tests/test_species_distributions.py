import errno
import scipy.sparse as sp
import numpy as np
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import SkipTest

#my tests below

def test_species_distributions_true():
    batch = fetch_species_distributions(data_home=None, download_if_missing=True)

def test_construct_grids():
    batch = fetch_species_distributions(data_home=None, download_if_missing=True)
    keep = construct_grids(batch)

#my tests above
