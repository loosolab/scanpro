import pytest

import numpy as np
import pandas as pd

from scanpro.get_transformed_props import get_transformed_props_counts
from scanpro.utils import del_index, pmax, pmin, is_fullrank, gauss_quad_prob, cov_to_corr
from scanpro.utils import matvec, vecmat, estimate_params_from_counts, estimate_beta_params
from scanpro.utils import norm_counts, simulate_cell_counts, simulate_cell_counts_2, convert_counts_to_df


@pytest.fixture()
def matrix_full_rank():
    return [[2, 3], [-1, 4]]


@pytest.fixture()
def matrix():
    return np.zeros((4, 5))


@pytest.fixture()
def cov_mat():
    return np.array([[1.0, 1.0, 8.1],
                     [1.0, 16.0, 18.0],
                     [8.1, 18.0, 81.0]])


@pytest.fixture()
def counts_matrix():
    """Simulated cell count matrix"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)

    return counts


@pytest.fixture()
def true_props():
    props_series = pd.DataFrame({'celltype': ['Cardiomyocytes', 'Endothelial cells', 'Epicardial cells',
                                              'Fibroblast', 'Immune cells', 'Neurons', 'Smooth muscle cells'],
                                 'props': [0.550716, 0.101838, 0.064313, 0.182517, 0.076476, 0.016180, 0.007960]}).set_index('celltype')

    return props_series


@pytest.fixture()
def beta_params():
    a = pd.Series({'Cardiomyocytes': 2.168850, 'Endothelial cells': 12.218475, 'Epicardial cells': 3.250916,
                   'Fibroblast': 3.156563, 'Immune cells': 1.361579, 'Neurons': 2.354012, 'Smooth muscle cells': 3.679757})

    b1 = pd.Series({'Cardiomyocytes': 1.769390, 'Endothelial cells': 107.760684, 'Epicardial cells': 47.297371,
                    'Fibroblast': 14.138098, 'Immune cells': 16.442437, 'Neurons': 143.134155, 'Smooth muscle cells': 458.574736})

    b2 = pd.Series({'Cardiomyocytes': 4.742325, 'Endothelial cells': 107.760684, 'Epicardial cells': 47.297371,
                    'Fibroblast': 4.666523, 'Immune cells': 16.442437, 'Neurons': 143.134155, 'Smooth muscle cells': 150.405074})

    b_grps = [b1, b2]

    return a, b_grps


def test_del_index():
    """Test del_index function"""
    values = np.array([list(range(5)) for _ in range(4)])
    exp = np.array([[0, 2, 4],
                    [0, 2, 4]])

    out = del_index(values, [1, 3])

    assert np.allclose(out, exp)


def test_pmax():
    """Test pmax function"""
    values = [1, 2, 3, 4, 5, 6]
    exp = [3, 3, 3, 4, 5, 6]

    out = pmax(values, 3)

    assert np.allclose(out, exp)


def test_pmin():
    """Test pmin function"""
    values = [1, 2, 3, 4, 5, 6]
    exp = [1, 2, 3, 3, 3, 3]

    out = pmin(values, 3)

    assert np.allclose(out, exp)


@pytest.mark.parametrize("value, rank", [('matrix', False), ('matrix_full_rank', True)])
def test_is_fullrank(value, rank, request):
    """Test is_fullrank function"""
    out = is_fullrank(request.getfixturevalue(value))

    if rank:
        assert out
    else:
        assert not out


def test_cov_to_corr(cov_mat):
    """Test cov_to_corr function"""
    exp = np.array([[1., 0.25, 0.9],
                    [0.25, 1., 0.5],
                    [0.9, 0.5, 1.]])

    out = cov_to_corr(cov_mat)

    assert np.allclose(out, exp)


def test_matvec(cov_mat):
    """Test matvec function"""
    exp = np.array([[1., 2., 8.1],
                    [1., 32., 18.],
                    [8.1, 36., 81.]])

    vec = np.array([1, 2, 1])

    out = matvec(cov_mat, vec)

    assert np.allclose(out, exp)


def test_vecmat(cov_mat):
    """Test vecmat function"""
    exp = np.array([[1., 1., 8.1],
                    [2., 32., 36.],
                    [8.1, 18., 81.]])

    vec = np.array([1, 2, 1])

    out = vecmat(vec, cov_mat)

    assert np.allclose(out, exp)


def test_gauss_quad_prob():
    """Test gauss_quad_prob function"""
    exp = np.array([[0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992],
                    [0.11846344, 0.23931434, 0.28444444, 0.23931434, 0.11846344]])

    out = gauss_quad_prob(5)

    assert np.allclose(out, exp)


def test_estimate_params_from_counts(counts_matrix):
    """Test estimate_params_from_counts function"""
    exp_a = np.array([45.13100692, 52.62544407, 38.38423792, 12.07542768])

    exp_b = np.array([132.00880094, 138.63188816, 153.14840101, 32.70428657])

    out = estimate_params_from_counts(counts_matrix)

    a = out[1].values
    b = out[2].values

    assert len(out) == 6
    assert np.allclose(a, exp_a)
    assert np.allclose(b, exp_b)


def test_estimate_beta_params(counts_matrix):
    """Test estimat_beta_params function"""
    props, _ = get_transformed_props_counts(counts_matrix)

    exp_a = np.array([0.76492002, 0.79233183, 0.78666146, 0.59952976])
    exp_b = np.array([3.05968009, 3.16932733, 3.14664585, 2.39811905])

    out = estimate_beta_params(props)

    a = out[0].values
    b = out[1].values

    assert np.allclose(a, exp_a)
    assert np.allclose(b, exp_b)


def test_norm_counts(counts_matrix):
    """Test norm_counts function"""
    exp = np.array([[961.0483871, 907.46307559, 794., 1065.53903598, 736.90238709],
                    [1158.91129032, 806.97219809, 953., 997.00950441, 906.20138967],
                    [664.25403226, 861.785404, 575., 779.04752206, 632.02342262],
                    [720.78629032, 928.77932233, 1183., 663.40393754, 1229.87280063]])

    out = norm_counts(counts_matrix)

    assert np.allclose(out.values, exp)


@pytest.mark.parametrize("n_reps", [1, 2, 3])
def test_simulate_cell_counts(n_reps):
    """Test simulate_cell_counts function"""
    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_clusters = len(p)
    n_samples = n_reps * 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)

    assert isinstance(counts, pd.DataFrame)
    assert len(counts.index) == n_samples
    assert len(counts.columns) == n_clusters


@pytest.mark.parametrize("n_reps", [1, 2, 3])
def test_simulate_cell_counts_2(n_reps, true_props, beta_params):
    """Test simulate_cell_counts_2 function"""
    a, b_grps = beta_params
    n_clusters = len(true_props)
    n_samples = n_reps * 2

    counts = simulate_cell_counts_2(true_props, n_reps,
                                    a, b_grps, n_conds=2, n=20, mu=5000)

    assert isinstance(counts, pd.DataFrame)
    assert len(counts.index) == n_samples
    assert len(counts.columns) == n_clusters


@pytest.mark.parametrize("n_reps, n_conds", [(2, 2), (3, 2), (2, 3)])
def test_convert_counts_to_df(n_reps, n_conds):
    """Test convert_counts_to_df function"""
    n_samples = n_reps * n_conds
    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p

    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=n_conds)
    df = convert_counts_to_df(counts, n_reps=n_reps, n_conds=n_conds)

    assert (column in df.columns for column in ['sample', 'cluster', 'group'])
    assert len(df['sample'].unique()) == n_samples
    assert len(df['group'].unique()) == n_conds
    assert df.index.name == 'cells'
