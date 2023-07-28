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
    counts = convert_counts_to_df(counts, column_name='cluster')

    return counts


@pytest.fixture
def counts_matrix():
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
    return np.array([[0.012891, 0.05649289, 0.15052133, 0.42445498, 0.35563981],
                     [0.01438092, 0.04647492, 0.16713434, 0.36741494, 0.40459488],
                     [0.01132257, 0.06817634, 0.13852084, 0.3943628, 0.38761744],
                     [0.00841029, 0.05029683, 0.19508575, 0.22988127, 0.51632586]])


@pytest.fixture
def trans_props_test():
    """Expected logit transformed proportions"""
    return np.array([[-4.33811208, -2.81546357, -1.73051439, -0.30452023, -0.59434586],
                     [-4.22725287, -3.02122251, -1.60607237, -0.54332916, -0.38636323],
                     [-4.46936679, -2.61502097, -1.82762389, -0.42901961, -0.45734783],
                     [-4.76966403, -2.93818151, -1.41729627, -1.20898315, 0.06531621]])


@pytest.fixture
def trans_props_norm_test():
    """Expected logit transformed proportions with normalized counts"""
    return np.array([[-1.29362251, -1.36615509, -1.53111778, -1.16011929, -1.62118045],
                     [-1.15089098, -1.60451909, -1.4011504, -1.34456474, -1.46352584],
                     [-1.45560766, -1.12344548, -1.63081786, -1.25511981, -1.516661],
                     [-1.71493395, -1.40811953, -1.09688514, -1.81215733, -1.04470727]])


@pytest.fixture
def trans_props_arcsin_test():
    """Expected arcsin transformed proportions"""
    return np.array([[0.11379174, 0.23998205, 0.39842955, 0.70956057, 0.63895161],
                     [0.12021661, 0.21728939, 0.42116182, 0.65120647, 0.68940259],
                     [0.10662015, 0.26417092, 0.3813619, 0.67895647, 0.67204485],
                     [0.09184532, 0.22619626, 0.45747613, 0.50003823, 0.80172431]])


@pytest.mark.parametrize("transform", ["logit", "arcsin"])
def test_get_transformed_props(counts_df, transform, props_test, trans_props_test, trans_props_arcsin_test):
    """Test get_transformed_props function"""
    counts, props, trans_props = get_transformed_props(counts_df, transform=transform,
                                                       sample_col='sample')

    # check if dataframes are produces
    assert all([isinstance(res, pd.DataFrame) for res in [counts, props, trans_props]])
    # check if dataframes have the right index and columns
    assert all(x in props.columns for x in [f'c{i}' for i in range(1, 6)])
    assert all(x in props.index for x in [f'S{i}' for i in range(1, 5)])
    # check if proportions are calcualted correctly
    assert np.allclose(props.values, props_test)
    # check if values are correct
    if transform == 'logit':
        assert np.allclose(trans_props.values, trans_props_test)
    if transform == 'arcsin':
        assert np.allclose(trans_props.values, trans_props_arcsin_test)


@pytest.mark.parametrize("transform, normalize", [("logit", True), ("arcsin", False)])
def test_get_transformed_props_counts(counts_matrix, transform, normalize, props_test,
                                      trans_props_norm_test, trans_props_arcsin_test):
    """Test get_transformed_props_counts function"""
    props, trans_props = get_transformed_props_counts(counts_matrix, transform=transform,
                                                      sample_col='sample', meta_cols=['group'],
                                                      normalize=normalize)
    # check if dataframes are produces
    assert all([isinstance(res, pd.DataFrame) for res in [props, trans_props]])
    # check if dataframes have the right index and columns
    assert all(x in props.columns for x in [f'c{i}' for i in range(1, 6)])
    assert all(x in props.index for x in [f'S{i}' for i in range(1, 5)])
    # check if proportions are calculated correctly
    assert np.allclose(props.values, props_test)
    # check if values are correct
    if transform == 'logit':
        assert np.allclose(trans_props.values, trans_props_norm_test)
    if transform == 'arcsin':
        assert np.allclose(trans_props.values, trans_props_arcsin_test)
