import pytest

import numpy as np
from statsmodels.tools.tools import add_constant

from scanpro.fitFDist import fit_f_dist, fit_f_dist_robust, trigamma_inverse, linkfun, linkinv, fun, winsorized_moments
from scanpro.get_transformed_props import get_transformed_props
from scanpro.linear_model import create_design, lm_fit
from scanpro.utils import simulate_cell_counts, convert_counts_to_df, gauss_quad_prob


@pytest.fixture()
def test_df():
    """Create a dummy dataframe with simulated samples, clusters and conditions"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=3)
    df = convert_counts_to_df(counts, n_reps=n_reps, n_conds=3)

    return df


@pytest.fixture()
def test_fit(test_df):
    """Create fit object"""
    # calculate proportions and transformed proportions
    _, props, prop_trans = get_transformed_props(test_df, sample_col='sample',
                                                 cluster_col='cluster', transform='logit')

    # create design matrix
    design = create_design(data=test_df, samples='sample', conds='group', reindex=props.index)

    coef = np.arange(len(design.columns))  # columns of the design matrix corresponding to conditions of interest

    # Change design matrix to intercept format
    design = design.iloc[:, 1:]
    design = add_constant(design, prepend=True, has_constant='skip')
    # fit fit linear model with all confounding variables
    fit = lm_fit(X=design, y=prop_trans)
    # remove intercept from stdev, coefficients and covariance matrix for the ebayes method
    fit['coefficients'] = fit['coefficients'][:, coef[1:]]
    fit['stdev'] = fit['stdev'][:, coef[1:]]
    fit['cov_coef'] = fit['cov_coef'][coef[1:][:, np.newaxis], coef[1:]]

    return fit


@pytest.fixture()
def test_params(test_fit):
    """Get sigma and residual degrees of freedom from fit object"""
    sigma = test_fit['sigma']
    df_residual = test_fit['df_residual']

    return sigma**2, df_residual


@pytest.fixture()
def g():
    """Get nodes and weights for gauss quad"""
    return gauss_quad_prob(128, dist='uniform')


@pytest.fixture()
def rbx():
    """Value used to test fun function"""
    return np.array([0.90710905])


@pytest.fixture()
def zwvar():
    """Value used to test fun function"""
    return 0.3485366766092676


@pytest.fixture()
def mom():
    """Value used to test fun function"""
    return np.array([-0.01586543, 0.32967972])


def test_linkfun():
    """Test linkfun function"""
    out = linkfun(1)
    assert out == 0.5


def test_linkinv():
    """Test linkinv function"""
    out = linkinv(2)
    assert out == -2


def test_fit_f_dist_robust(test_params):
    """Test fit_f_dist_robust function"""
    var, df = test_params

    out = fit_f_dist_robust(var, df, covariate=None, winsor_tail_p=[0.05, 0.1])

    assert all([stat in out.keys() for stat in ['scale', 'df2', 'df2_shrunk']])
    assert np.isclose(0.11209644570775239, out['scale'])
    assert np.isinf(out['df2'])
    assert all(np.isinf(out['df2_shrunk']))


def test_fit_f_dist(test_params):
    """Test fit_f_dist function"""
    var, df = test_params

    out = fit_f_dist(var, df, covariate=None)

    assert all([stat in out.keys() for stat in ['scale', 'df2']])
    assert np.isclose(0.11209644570775239, out['scale'])
    assert np.isinf(out['df2'])


def test_trigamma_inverse():
    """Test trigamma_inverse function"""
    out = trigamma_inverse(1)

    assert all(np.isclose(1.42625512, out))


def test_fun(rbx, mom, zwvar, g):
    """Test fun function"""
    winsor_tail_p = np.array([0.05, 0.1])
    value = np.log(zwvar / mom[1])

    out = fun(rbx, 10, linkinv, winsorized_moments, zwvar, winsor_tail_p, linkfun, g)

    assert np.isclose(out, value)


def test_winsorized_moments(g):
    """Test winsorized_moments function"""
    winsor_tail_p = np.array([0.05, 0.1])

    out = winsorized_moments(10, 1e10, winsor_tail_p, linkfun, linkinv, g)

    assert all(np.isclose(out, [-0.10712624, 0.16791501]))
