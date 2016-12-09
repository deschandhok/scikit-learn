from sklearn.utils.testing import assert_array_equal, assert_raises

from sklearn.utils.stats import rankdata

from sklearn.utils.stats import _rankdata

import numpy as np


_cases = (
    # values, method, expected
    ([100], 'max', [1.0]),
    ([100, 100, 100], 'max', [3.0, 3.0, 3.0]),
    ([100, 300, 200], 'max', [1.0, 3.0, 2.0]),
    ([100, 200, 300, 200], 'max', [1.0, 3.0, 4.0, 3.0]),
    ([100, 200, 300, 200, 100], 'max', [2.0, 4.0, 5.0, 4.0, 2.0]),
)


def test_cases():

    def check_case(values, method, expected):
        r = rankdata(values, method=method)
        assert_array_equal(r, expected)

    for values, method, expected in _cases:
        yield check_case, values, method, expected

#mycode below

#raise NotImplementedError()
#seeded fault is line 40
def test_not_implemented():
   assert_raises(NotImplementedError, _rankdata, np.array([2,10,8,17]), method="average")

#seeded fault is line 51
def test_rank_rankdata():
   r = _rankdata (np.array([2,10,8,17]), method="max")
   assert_array_equal(r, np.array([1, 3, 2, 4]))

#mycode above
