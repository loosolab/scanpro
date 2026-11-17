import pytest
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scanpro import scanpro, run_scanpro, run_stats, anova, t_test, sim_scanpro
from scanpro.get_transformed_props import get_transformed_props
from scanpro.linear_model import create_design
from scanpro.utils import simulate_cell_counts, convert_counts_to_df
from scanpro.result import ScanproResult

matplotlib.use("Agg")  # non-interactive backend for tests


@pytest.fixture
def counts_df():
    """A simulated cell count dataframe with five clusters and two conditions"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=2)
    counts = convert_counts_to_df(counts, column_name='cluster')
    counts['merged_samples'] = counts['group'].astype(str)

    return counts


@pytest.fixture
def counts_df_3():
    """A simulated cell count dataframe with five clusters and two conditions"""
    np.random.seed(10)

    p = np.array([0.01, 0.05, 0.15, 0.34, 0.45])  # true clusters proportions
    a = 10
    b = a * (1 - p) / p
    n_reps = 2
    counts = simulate_cell_counts(props=p, n_reps=n_reps, a=a, b=b, n_conds=3)
    counts = convert_counts_to_df(counts, column_name='cluster')

    return counts


def test_import():
    """ Test if scanpro is imported correctly """

    assert scanpro.__name__ == "scanpro"


@pytest.mark.parametrize("transform, samples, pairwise", [("logit", None, False),
                                                          ("arcsin", None, False),
                                                          ('logit', 'sample', False),
                                                          ('arcsin', 'sample', True)])
def test_scanpro(counts_df, transform, samples, pairwise):
    """Test scanpro wrapper function"""
    out = scanpro(counts_df, 'cluster', 'group', samples_col=samples,
                  transform=transform, pairwise=pairwise, verbosity=0)

    assert isinstance(out, ScanproResult) and isinstance(out.results, pd.DataFrame)
    if samples is None:
        assert isinstance(out.sim_results, pd.DataFrame)
        assert "adjusted_p_values" in out.results.columns and "adjusted_p_values" in out.sim_results.columns
    else:
        if not pairwise:
            assert all(x in out.results.columns for x in ['p_values', 'adjusted_p_values'])
        else:
            assert all(x in out.results.columns for x in [f'adjusted_p_values_{pair[0]}_{pair[1]}' for pair in out.condition_pairs])


@pytest.mark.parametrize("transform, conditions", [("logit", None),
                                                   ("arcsin", None),
                                                   ('logit', ['cond_1', 'cond_2']),
                                                   ('arcsin', ['cond_1', 'cond_2'])])
def test_run_scanpro(counts_df_3, transform, conditions):
    """Test run_scanpro function"""
    if not conditions:
        conditions = counts_df_3['group'].unique().tolist()
    out = run_scanpro(counts_df_3, clusters_col='cluster', samples_col='sample',
                      conds_col='group', conditions=conditions,
                      transform=transform, verbosity=0)

    assert isinstance(out, ScanproResult) and isinstance(out.results, pd.DataFrame)
    assert all(x in out.results.columns for x in ['p_values', 'adjusted_p_values'])


@pytest.mark.parametrize("transform, conditions", [("logit", None),
                                                   ("arcsin", None),
                                                   ('logit', ['cond_1', 'cond_2']),
                                                   ('arcsin', ['cond_1', 'cond_2'])])
def test_run_stats(counts_df_3, transform, conditions):
    """Test run_stats function"""
    out = run_stats(counts_df_3, clusters='cluster', samples='sample',
                    conds='group', conditions=conditions,
                    transform=transform, verbosity=0)

    assert isinstance(out, ScanproResult) and isinstance(out.results, pd.DataFrame)
    assert all(x in out.results.columns for x in ['p_values', 'adjusted_p_values'])


@pytest.mark.parametrize("transform", ["logit", "arcsin"])
def test_anova(counts_df_3, transform):
    """Test anova function."""
    # calculate proportions and transformed proportions
    counts, props, prop_trans = get_transformed_props(counts_df_3, sample_col='sample',
                                                      cluster_col='cluster', transform=transform)
    # create design matrix
    design = create_design(data=counts_df_3, sample_col='sample',
                           conds_col='group')

    coef = np.arange(len(design.columns))

    # run anova
    out = anova(props, prop_trans, design, coef, verbosity=0)

    assert isinstance(out, pd.DataFrame)
    assert all(x in out.columns for x in ['p_values', 'adjusted_p_values'])


@pytest.mark.parametrize("transform", ["logit", "arcsin"])
def test_t_test(counts_df, transform):
    """Test t_test function."""
    # calculate proportions and transformed proportions
    counts, props, prop_trans = get_transformed_props(counts_df, sample_col='sample',
                                                      cluster_col='cluster', transform=transform)
    # create design matrix
    design = create_design(data=counts_df, sample_col='sample',
                           conds_col='group')

    contrasts = [1, -1]
    # run anova
    out = t_test(props, prop_trans, design, contrasts, verbosity=0)

    assert isinstance(out, pd.DataFrame)
    assert all(x in out.columns for x in ['p_values', 'adjusted_p_values'])


def test_sim_scanpro(counts_df):
    """Test sim_scanpro function"""
    out = sim_scanpro(counts_df, 'cluster', 'group',
                      transform='arcsin', n_reps=8, n_sims=100,
                      conditions=['cond_1', 'cond_2'],
                      robust=True, verbosity=0)

    assert isinstance(out, ScanproResult) and isinstance(out.results, pd.DataFrame)
    assert "adjusted_p_values" in out.results.columns


@pytest.mark.parametrize("kind", ["stripplot", "boxplot", "barplot"])
@pytest.mark.parametrize("samples, pairwise", [(None, False),
                                               ('sample', False),
                                               (None, True),
                                               ('sample', True)])
def test_plot(counts_df, samples, pairwise, kind):
    """Test plotting of scanpro results with replicated and unreplicated data"""
    out = scanpro(counts_df, 'cluster', 'group', samples_col=samples,
                  pairwise=pairwise, n_sims=10, verbosity=0)

    axes = out.plot(kind=kind, show=False)
    plt.close('all')

    assert len(axes) > 0
