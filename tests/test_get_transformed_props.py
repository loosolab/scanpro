import pytest

import numpy as np
import pandas as pd
from scanpro.get_transformed_props import get_transformed_props, get_transformed_props_counts
from scanpro.utils import simulate_cell_counts, convert_counts_to_df


@pytest.fixture
def counts_df():
    """Dataframe containing simulated cell counts"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)
    counts_df = convert_counts_to_df(counts, n_reps=n_reps, n_conds=2)

    return counts_df


@pytest.fixture
def counts():
    """Simulated cell count matrix"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)

    return counts


@pytest.fixture
def props_test():
    """Expected proportions for simulated cell """
    return np.array([[0.01289277, 0.05649425, 0.1505218 , 0.42445285, 0.35563834],
                     [0.01438255, 0.04647627, 0.16713463, 0.36741347, 0.40459308],
                     [0.01132485, 0.06817793, 0.13852158, 0.39436046, 0.38761518],
                     [0.00841187, 0.05029807, 0.19508579, 0.22988102, 0.51632325]])


@pytest.fixture
def trans_props_test():
    """Expected logit transformed proportions"""
    return np.array([[-4.33130943, -2.81421441, -1.73033476, -0.30494678, -0.59466117],
                     [-4.22164446, -3.01973566, -1.60597098, -0.54363854, -0.386728  ],
                     [-4.45947489, -2.61379766, -1.82732004, -0.42949965, -0.45781407],
                     [-4.76042982, -2.93691657, -1.41728364, -1.20905132,  0.06480467]])


@pytest.fixture
def trans_props_norm_test():
    """Expected logit transformed proportions with normalized counts"""
    return np.array([[-1.29365443, -1.36616217, -1.53106414, -1.16019453, -1.62109072],
                     [-1.15096333, -1.60444235, -1.40114551, -1.34457824, -1.46349993],
                     [-1.45557582, -1.12355567, -1.6306987 , -1.25517664, -1.51659992],
                     [-1.71481121, -1.40811218, -1.09697477, -1.81199221, -1.04481185]])


@pytest.fixture
def trans_props_arcsin_test():
    """Expected arcsin transformed proportions"""
    return np.array([[0.11379174, 0.23998205, 0.39842955, 0.70956057, 0.63895161],
                     [0.12021661, 0.21728939, 0.42116182, 0.65120647, 0.68940259],
                     [0.10662015, 0.26417092, 0.3813619 , 0.67895647, 0.67204485],
                     [0.09184532, 0.22619626, 0.45747613, 0.50003823, 0.80172431]])


@pytest.mark.parametrize("transform", ["logit", "arcsin"])
def test_get_transformed_props(counts_df, transform, props_test, trans_props_test, trans_props_arcsin_test):
    """Test get_transformed_props function"""
    counts, props, trans_props = get_transformed_props(counts_df, transform=transform)

    # check if dataframes are produces
    assert all([isinstance(res, pd.DataFrame) for res in [counts, props, trans_props]])
    # check if dataframes have the right index and columns
    assert all(x in props.columns for x in [f'c{i}' for i in range(1, 6)])
    assert all(x in props.index for x in [f'S{i}' for i in range(1, 6)])
    # check if proportions are calcualted correctly
    assert np.allclose(props.values, props_test)
    # check if values are correct
    if transform == 'logit':
        assert np.allclose(trans_props.values, trans_props_test)
    if transform == 'arcsin':
        assert np.allclose(trans_props.values, trans_props_arcsin_test)


@pytest.mark.parametrize("transform, normalize", [("logit", True), ("arcsin", False)])
def test_get_transformed_props_counts(counts, transform, normalize, props_test,
                                      trans_props_norm_test, trans_props_arcsin_test):
    """Test get_transformed_props_counts function"""
    props, trans_props = get_transformed_props_counts(counts, transform=transform, normalize=normalize)

    # check if dataframes are produces
    assert all([isinstance(res, pd.DataFrame) for res in [counts, props, trans_props]])
    # check if dataframes have the right index and columns
    assert all(x in props.columns for x in [f'c{i}' for i in range(1, 6)])
    assert all(x in props.index for x in [f'S{i}' for i in range(1, 6)])
    # check if proportions are calculated correctly
    assert np.allclose(props.values, props_test)
    # check if values are correct
    if transform == 'logit':
        assert np.allclose(trans_props.values, trans_props_norm_test)
    if transform == 'arcsin':
        assert np.allclose(trans_props.values, trans_props_arcsin_test)
