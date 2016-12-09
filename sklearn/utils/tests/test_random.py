from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.misc import comb as combinations
from numpy.testing import assert_array_almost_equal
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.random import random_choice_csc
from sklearn.utils.random import choice

from sklearn.utils.testing import (
    assert_raises,
    assert_equal,
    assert_true)


###############################################################################
# test custom sampling without replacement algorithm
###############################################################################

#stuff I did is below
def test_choice_a_n_dimensional():
   y = np.zeros((2, 3, 4))
   assert_raises(ValueError, choice, y)

def test_choice_a_zero_dimensional_pop_size_less_than_zero():
   assert_raises(ValueError, choice, -2)


def test_choice_a_one_dimensional_pop_size():
   assert_raises(ValueError, choice, np.array([]))


def test_choice_a_zero_dimensional_pop_size():
   assert_raises(ValueError, choice, 's')



def test_choice_p_not_Non_1_dimensional():
   assert_raises(ValueError, choice, 3, p=np.zeros((0.2, 0.7, 0.1)))
   

def test_choice_p_size_not_equal_pop_size():
   assert_raises(ValueError, choice, 3, p=np.array([0.6, 0.1]))

def test_choice_p_less_than_zero():
   assert_raises(ValueError, choice, 3, p=np.array([-0.6, 0.1, 0.3]))

def test_choice_p_not_equal_one():
   assert_raises(ValueError, choice, 3, p=np.array([0.9, 0.1, 0.3]))


def test_returns_1d_scalar_object_based_off_distribution():
   keeper = choice(3, replace=False, p= np.array([0,0,1.0]))
   assert_equal(keeper, 2)


#if np.sum(p > 0) < size: 
#raise ValueError("Fewer non-zero entries in p than size") 
def test_sum_more_p_zero_than_size():
   assert_raises(ValueError, choice, 4, size=3, replace= False, p=np.array([0, 0, 0.5, 0.5]))


#if size > pop_size: 
#raise ValueError("Cannot take a larger sample than ""population when 'replace=False'")
def test_size_greater_than_pop_size():
   assert_raises(ValueError, choice, 4, size=5, replace= False)


# idx = random_state.randint(0, pop_size, size=shape) 
def test_non_p_array():
   keeper = choice(np.array(1), size=())
   assert_equal(keeper[()], 0)


#if shape is not None and idx.ndim == 0: 

# If size == () then the user requested a 0-d array as opposed to 
# a scalar object when size is None. However a[idx] is always a 
# scalar and not an array. So this makes sure the result is an 
# array, taking into account that np.array(item) may not work 
# for object arrays. 
def test_0_d_array():
   keeper = choice(4, size=(), p=np.array([0, 0, 0.0, 1.0]))
   assert_equal(keeper[()], 3)


#stuff I did is above




def test_invalid_sample_without_replacement_algorithm():
    assert_raises(ValueError, sample_without_replacement, 5, 4, "unknown")


def test_sample_without_replacement_algorithms():
    methods = ("auto", "tracking_selection", "reservoir_sampling", "pool")

    for m in methods:
        def sample_without_replacement_method(n_population, n_samples,
                                              random_state=None):
            return sample_without_replacement(n_population, n_samples,
                                              method=m,
                                              random_state=random_state)

        check_edge_case_of_sample_int(sample_without_replacement_method)
        check_sample_int(sample_without_replacement_method)
        check_sample_int_distribution(sample_without_replacement_method)


def check_edge_case_of_sample_int(sample_without_replacement):

    # n_population < n_sample
    assert_raises(ValueError, sample_without_replacement, 0, 1)
    assert_raises(ValueError, sample_without_replacement, 1, 2)

    # n_population == n_samples
    assert_equal(sample_without_replacement(0, 0).shape, (0, ))

    assert_equal(sample_without_replacement(1, 1).shape, (1, ))

    # n_population >= n_samples
    assert_equal(sample_without_replacement(5, 0).shape, (0, ))
    assert_equal(sample_without_replacement(5, 1).shape, (1, ))

    # n_population < 0 or n_samples < 0
    assert_raises(ValueError, sample_without_replacement, -1, 5)
    assert_raises(ValueError, sample_without_replacement, 5, -1)


def check_sample_int(sample_without_replacement):
    # This test is heavily inspired from test_random.py of python-core.
    #
    # For the entire allowable range of 0 <= k <= N, validate that
    # the sample is of the correct length and contains only unique items
    n_population = 100

    for n_samples in range(n_population + 1):
        s = sample_without_replacement(n_population, n_samples)
        assert_equal(len(s), n_samples)
        unique = np.unique(s)
        assert_equal(np.size(unique), n_samples)
        assert_true(np.all(unique < n_population))

    # test edge case n_population == n_samples == 0
    assert_equal(np.size(sample_without_replacement(0, 0)), 0)


def check_sample_int_distribution(sample_without_replacement):
    # This test is heavily inspired from test_random.py of python-core.
    #
    # For the entire allowable range of 0 <= k <= N, validate that
    # sample generates all possible permutations
    n_population = 10

    # a large number of trials prevents false negatives without slowing normal
    # case
    n_trials = 10000

    for n_samples in range(n_population):
        # Counting the number of combinations is not as good as counting the
        # the number of permutations. However, it works with sampling algorithm
        # that does not provide a random permutation of the subset of integer.
        n_expected = combinations(n_population, n_samples, exact=True)

        output = {}
        for i in range(n_trials):
            output[frozenset(sample_without_replacement(n_population,
                                                        n_samples))] = None

            if len(output) == n_expected:
                break
        else:
            raise AssertionError(
                "number of combinations != number of expected (%s != %s)" %
                (len(output), n_expected))


def test_random_choice_csc(n_samples=10000, random_state=24):
    # Explicit class probabilities
    classes = [np.array([0, 1]),  np.array([0, 1, 2])]
    class_probabilites = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]

    got = random_choice_csc(n_samples, classes, class_probabilites,
                            random_state)
    assert_true(sp.issparse(got))

    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel()) / float(n_samples)
        assert_array_almost_equal(class_probabilites[k], p, decimal=1)

    # Implicit class probabilities
    classes = [[0, 1],  [1, 2]]  # test for array-like support
    class_probabilites = [np.array([0.5, 0.5]), np.array([0, 1/2, 1/2])]

    got = random_choice_csc(n_samples=n_samples,
                            classes=classes,
                            random_state=random_state)
    assert_true(sp.issparse(got))

    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel()) / float(n_samples)
        assert_array_almost_equal(class_probabilites[k], p, decimal=1)

    # Edge case probabilities 1.0 and 0.0
    classes = [np.array([0, 1]),  np.array([0, 1, 2])]
    class_probabilites = [np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0])]

    got = random_choice_csc(n_samples, classes, class_probabilites,
                            random_state)
    assert_true(sp.issparse(got))

    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel(),
                        minlength=len(class_probabilites[k])) / n_samples
        assert_array_almost_equal(class_probabilites[k], p, decimal=1)

    # One class target data
    classes = [[1],  [0]]  # test for array-like support
    class_probabilites = [np.array([0.0, 1.0]), np.array([1.0])]

    got = random_choice_csc(n_samples=n_samples,
                            classes=classes,
                            random_state=random_state)
    assert_true(sp.issparse(got))

    for k in range(len(classes)):
        p = np.bincount(got.getcol(k).toarray().ravel()) / n_samples
        assert_array_almost_equal(class_probabilites[k], p, decimal=1)


def test_random_choice_csc_errors():
    # the length of an array in classes and class_probabilites is mismatched
    classes = [np.array([0, 1]),  np.array([0, 1, 2, 3])]
    class_probabilites = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    assert_raises(ValueError, random_choice_csc, 4, classes,
                  class_probabilites, 1)

    # the class dtype is not supported
    classes = [np.array(["a", "1"]),  np.array(["z", "1", "2"])]
    class_probabilites = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    assert_raises(ValueError, random_choice_csc, 4, classes,
                  class_probabilites, 1)

    # the class dtype is not supported
    classes = [np.array([4.2, 0.1]),  np.array([0.1, 0.2, 9.4])]
    class_probabilites = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    assert_raises(ValueError, random_choice_csc, 4, classes,
                  class_probabilites, 1)

    # Given probabilities don't sum to 1
    classes = [np.array([0, 1]),  np.array([0, 1, 2])]
    class_probabilites = [np.array([0.5, 0.6]), np.array([0.6, 0.1, 0.3])]
    assert_raises(ValueError, random_choice_csc, 4, classes,
                  class_probabilites, 1)
