import pytest

import numpy as np
import pandas as pd

from scanpro.get_transformed_props import get_transformed_props
from scanpro.linear_model import create_design, lm_fit, contrasts_fit
from scanpro.utils import simulate_cell_counts, convert_counts_to_df


@pytest.fixture
def counts_df():
    """Create a dummy dataframe with simulated samples, clusters and conditions"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)
    counts = convert_counts_to_df(counts, column_name='cluster')

    return counts


def test_create_design(counts_df):
    """Test create_design function."""
    # calculate proportions and transformed proportions
    _, props, _ = get_transformed_props(counts_df, sample_col='sample',
                                        cluster_col='cluster')
    # create design matrix
    design = create_design(data=counts_df, samples='sample',
                           conds='group', reindex=props.index)

    assert isinstance(design, pd.DataFrame)
    # check columns -> conditions
    assert all(x in design.columns for x in counts_df['group'].unique())
    # check rows -> samples
    assert all(x in design.index for x in counts_df['sample'].unique())


def test_lm_fit(counts_df):
    """test lm_fit function"""
    # calculate proportions and transformed proportions
    _, props, prop_trans = get_transformed_props(counts_df, sample_col='sample',
                                                 cluster_col='cluster')
    # create design matrix
    design = create_design(data=counts_df, samples='sample',
                           conds='group', reindex=props.index)
    # run lm_fit function
    fit = lm_fit(design, prop_trans)

    # check if output is dictionary
    assert isinstance(fit, dict)
    # check if all keys are available
    stats = ['coefficients', 'sigma', 'stdev', 'df_residual', 'ssr', 'design', 'cov_coef']
    assert all([stat in fit.keys() for stat in stats])
    # check if all clusters have coefficients
    assert len(fit['coefficients']) == len(counts_df['cluster'].unique())


def test_contrasts_fit(counts_df):
    """Test contrasts_fit function"""
    # calculate proportions and transformed proportions
    _, props, prop_trans = get_transformed_props(counts_df, sample_col='sample',
                                                 cluster_col='cluster')
    # create design matrix
    design = create_design(data=counts_df, samples='sample',
                           conds='group', reindex=props.index)
    # run lm_fit function
    fit = lm_fit(design, prop_trans)
    # run contrasts_fit function
    fit_cont = contrasts_fit(fit, contrasts=[1, -1])

    # check if output is dictionary
    assert isinstance(fit_cont, dict)
    # check if all keys are available
    stats = ['coefficients', 'sigma', 'stdev', 'df_residual', 'ssr', 'design', 'cov_coef']
    assert all([stat in fit_cont.keys() for stat in stats])
    # check if all clusters have coefficients
    assert len(fit['coefficients']) == len(counts_df['cluster'].unique())
