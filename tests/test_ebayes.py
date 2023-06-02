import pytest

import numpy as np
import pandas as pd
from statsmodels.tools.tools import add_constant

from scanpro.ebayes import ebayes, squeeze_var, tmixture_vector, tmixture_matrix, classify_tests_f
from scanpro.get_transformed_props import get_transformed_props
from scanpro.linear_model import create_design, lm_fit
from scanpro.utils import simulate_cell_counts, convert_counts_to_df, pmin, pmax


@pytest.fixture()
def test_df():
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


def test_ebayes(test_fit):
    """Test ebayes function"""
    out = ebayes(test_fit, robust=True)

    assert isinstance(out, dict)

    fit_keys = ['df_prior', 's2_prior', 'var_prior', 'proportion',
                's2_post', 'p_value', 't', 'df_total', 'lods', 'F']

    assert all([x in out.keys() for x in fit_keys])


def test_squeeze_var(test_fit):
    """Test squeeze var function"""
    sigma = test_fit['sigma']  # sigma
    df_residual = test_fit['df_residual']    
    winsor_tail_p = [0.05, 0.1]

    var_prior, var_post, df_prior = squeeze_var(sigma**2, df_residual, robust=True, winsor_tail_p=winsor_tail_p)

    assert all([isinstance(stat, np.ndarray) for stat in [var_prior, var_post, df_prior]])
    n_clusters = 5  # number of clusters in simulated data
    assert all([len(stat) == n_clusters for stat in [var_prior, var_post, df_prior]])


def test_tmixture_matrix(test_fit):
    """Test tmixture_matrix function"""
    stdev_coef_lim = np.array([0.1, 4])
    coefficients = test_fit['coefficients']  # beta
    stdev = test_fit['stdev']  # standard deviation (stdev.unscaled in R's version)
    sigma = test_fit['sigma']  # sigma
    df_residual = test_fit['df_residual']    
    winsor_tail_p = [0.05, 0.1]
    proportion=0.01
    n_clusters = 5

    var_prior, var_post, df_prior = squeeze_var(sigma**2, df_residual, robust=True, winsor_tail_p=winsor_tail_p)
   # calcualte t-staisctics
    t_stat = coefficients / stdev / np.reshape(np.sqrt(var_post), (n_clusters, 1))
    df_total = df_residual + df_prior
    df_pooled = np.nansum(df_residual)
    df_total = pmin(df_total, df_pooled)
    var_prior_lim = stdev_coef_lim**2 / var_prior

    out = tmixture_matrix(t_stat, stdev, df_total, proportion, var_prior_lim)

    assert isinstance(out, np.ndarray)
    # check if length of output == number of conditions
    n_conds = 3
    assert len(out) == n_conds


def test_tmixture_vector(test_fit):
    """Test tmixture_vector function"""
    stdev_coef_lim = np.array([0.1, 4])
    coefficients = test_fit['coefficients']  # beta
    stdev = test_fit['stdev']  # standard deviation (stdev.unscaled in R's version)
    sigma = test_fit['sigma']  # sigma
    df_residual = test_fit['df_residual']    
    winsor_tail_p = [0.05, 0.1]
    proportion=0.01
    n_clusters = 5

    var_prior, var_post, df_prior = squeeze_var(sigma**2, df_residual, robust=True, winsor_tail_p=winsor_tail_p)
   # calcualte t-staisctics
    t_stat = coefficients / stdev / np.reshape(np.sqrt(var_post), (n_clusters, 1))
    df_total = df_residual + df_prior
    df_pooled = np.nansum(df_residual)
    df_total = pmin(df_total, df_pooled)
    var_prior_lim = stdev_coef_lim**2 / var_prior
    v0 = tmixture_vector(t_stat[:, 0], stdev[:, 0], df_total, proportion, var_prior_lim)

    assert isinstance(v0, float)


def test_classify_tests_f(test_fit):
    """Test classify_tests_f function"""
    out = classify_tests_f(test_fit)

    assert isinstance(out, dict)
    # check if all results are in the dictionary
    assert all([x in out.keys() for x in ['stat', 'df1', 'df2']])
