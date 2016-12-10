import errno
import scipy.sparse as sp
import numpy as np
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true, assert_raises
from sklearn.utils.testing import SkipTest

#my tests below

def test_species_distributions_true():
    batch = fetch_species_distributions(data_home=None, download_if_missing=True)  

    assert_equal(batch.coverages.shape, (14, 1592, 1212))
    assert_equal(batch.train.shape, (1624,))
    assert_equal(batch.test.shape, (620,))


def test_construct_grids():
    batch = fetch_species_distributions(data_home=None, download_if_missing=True)
    keep = construct_grids(batch)

   
    xmin = batch.x_left_lower_corner + batch.grid_size  
    xmax = xmin + (batch.Nx * batch.grid_size) 

    ymin = batch.y_left_lower_corner + batch.grid_size 
    ymax = ymin + (batch.Ny * batch.grid_size)

    xgrid = np.arange(xmin, xmax, batch.grid_size) 
    ygrid = np.arange(ymin, ymax, batch.grid_size) 

    assert_array_equal(keep[0], xgrid)
    assert_array_equal(keep[1], ygrid)


#my tests above
