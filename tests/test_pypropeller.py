import pytest

import numpy as np
import pandas as pd
from pypropeller import pypropeller
from pypropeller.get_transformed_props import get_transformed_props
from pypropeller.linear_model import create_design
from pypropeller.utils import simulate_cell_counts, convert_counts_to_df
from pypropeller.result import PyproResult


@pytest.fixture
def counts_df():
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)
    counts_df = convert_counts_to_df(counts, n_reps=n_reps, n_conds=2)

    return counts_df


@pytest.fixture
def counts_df_3():
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=3)
    counts_df_3 = convert_counts_to_df(counts, n_reps=n_reps, n_conds=3)

    return counts_df_3


def test_import():
    """ Test if pypropeller is imported correctly """

    assert pypropeller.__name__ == "pypropeller.pypropeller"


@pytest.mark.parametrize("transform, samples", [("logit", None),
                                                ("arcsin", None),
                                                ('logit', 'sample'),
                                                ('arcsin', 'sample')])
def test_pypropeller(counts_df, transform, samples):
    """Test pypropeller wrapper function"""
    out = pypropeller.pypropeller(counts_df, 'cluster', 'group', samples_col=samples,
                      transform=transform, verbose=False)
    
    assert isinstance(out, PyproResult) and isinstance(out.results, pd.DataFrame)
    if samples is None:
        assert isinstance(out.sim_results, pd.DataFrame)
        assert "p_values" in out.results.columns and "p_values" in out.sim_results.columns
    else:
        assert all(x in out.results.columns for x in ['p_values', 'Adjusted_p_values'])


@pytest.mark.parametrize("transform, conditions", [("logit", None),
                                                  ("arcsin", None),
                                                  ('logit', ['cond_1', 'cond_2']),
                                                  ('arcsin', ['cond_1', 'cond_2'])])
def test_run_pypropeller(counts_df_3, transform, conditions):
    """Test run_pypropeller function"""
    out = pypropeller.run_pypropeller(counts_df_3, clusters='cluster', samples='sample',
                                      conds='group', conditions=conditions,
                                      transform=transform, verbose=False)

    assert isinstance(out, PyproResult) and isinstance(out.results, pd.DataFrame)
    assert all(x in out.results.columns for x in ['p_values', 'Adjusted_p_values'])


@pytest.mark.parametrize("transform", ["logit", "arcsin"])
def test_anova(counts_df_3, transform):
    """Test anova function."""
    # calculate proportions and transformed proportions
    counts, props, prop_trans = get_transformed_props(counts_df_3, sample_col='sample',
                                                      cluster_col='cluster', transform=transform)
    # create design matrix
    design = create_design(data=counts_df_3, samples='sample',
                           conds='group', reindex=props.index)

    coef = np.arange(len(design.columns))

    # run anova
    out = pypropeller.anova(props, prop_trans, design, coef, verbose=False)

    assert isinstance(out, pd.DataFrame)
    assert all(x in out.columns for x in ['p_values', 'Adjusted_p_values'])


@pytest.mark.parametrize("transform", ["logit", "arcsin"])
def test_t_test(counts_df, transform):
    """Test t_test function."""
    # calculate proportions and transformed proportions
    counts, props, prop_trans = get_transformed_props(counts_df, sample_col='sample',
                                                      cluster_col='cluster', transform=transform)
    # create design matrix
    design = create_design(data=counts_df, samples='sample',
                           conds='group', reindex=props.index)

    contrasts = [1, -1]
    # run anova
    out = pypropeller.t_test(props, prop_trans, design, contrasts, verbose=False)

    assert isinstance(out, pd.DataFrame)
    assert all(x in out.columns for x in ['p_values', 'Adjusted_p_values'])


@pytest.mark.parametrize("data", [counts_df, counts_df_3])
def test_sim_pypropeller(data):
    """Test sim_pypropeller function"""
    out = pypropeller.sim_pypropeller(data, 'cluster', 'group', samples_col=None,
                                      transform='arcsin', n_reps=8, n_sims=100,
                                      conditions=None, robust=True, verbose=False)
    
    assert isinstance(out, PyproResult) and isinstance(out.results, pd.DataFrame)
    assert "p_values" in out.results.columns
