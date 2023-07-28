import pytest

import numpy as np
from scanpro.scanpro import run_scanpro
from scanpro.sim_reps import generate_reps, combine, get_mean_sim
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
    counts_df = convert_counts_to_df(counts, column_name='cluster')

    return counts_df


@pytest.fixture
def coefficients():
    """Coefficients of two conditions to test combine function"""
    fit = {'cond_1': np.array([[0.01860629, 0.04741851, 0.16473536, 0.39056604, 0.37867379],
                               [0.01434144, 0.05097195, 0.15868627, 0.39802525, 0.37797508],
                               [0.01536439, 0.05282983, 0.1555466, 0.39024898, 0.3860102],
                               [0.01350291, 0.0429639, 0.17366848, 0.35909548, 0.41076923],
                               [0.01173468, 0.05070031, 0.15823847, 0.39373346, 0.38559308]]),
           'cond_2': np.array([[0.01499118, 0.07476602, 0.17289819, 0.28538108, 0.45196353],
                               [0.0067029, 0.07274286, 0.16928989, 0.29888145, 0.4523829],
                               [0.01356278, 0.05925851, 0.16602386, 0.30615178, 0.45500307],
                               [0.00863847, 0.05860963, 0.17158782, 0.29647629, 0.46468779],
                               [0.00752065, 0.04287439, 0.18210656, 0.2517275, 0.5157709]])}
    return fit


@pytest.fixture
def counts_list(counts_df):
    """List of simulated counts matrices as pandas dataframes"""
    np.random.seed(10)

    n_sims = 3
    n_reps = 2
    samples_col = 'sample'
    rep_samples_col = 'sample_replicates'
    conds_col = 'group'
    clusters_col = 'cluster'
    transform = 'arcsin'
    # copy dataframe
    counts_df = counts_df.copy()
    # add conds_col as samples_col
    counts_df[samples_col] = counts_df[conds_col]
    # initiate list to save counts
    counts = []
    for i in range(n_sims):
        # generate replicates
        rep_data = generate_reps(data=counts_df, n_reps=n_reps, sample_col=samples_col)

        # run propeller
        try:
            out_sim = run_scanpro(rep_data, clusters=clusters_col, samples=rep_samples_col,
                                  conds=conds_col, transform=transform,
                                  conditions=None, robust=True, verbosity=0)
        # workaround brentq error "f(a) and f(b) must have different signs"
        # rerun simulation instead of crashing
        except ValueError:
            i -= 1
            continue

        # save counts, props and prop_trans
        counts.append(out_sim.counts)

    return counts


@pytest.fixture
def exp_mean_counts():
    """Expected output of get_mean_sim using counts_list"""
    exp_values = np.array([[23., 103.66666667, 274., 784., 644.33333333],
                           [15.66666667, 74., 199., 538.66666667, 449.],
                           [23., 79.66666667, 285.33333333, 639.33333333, 677.33333333],
                           [14.66666667, 53.66666667, 197.66666667, 415., 465.66666667],
                           [25.33333333, 154.33333333, 310.33333333, 895.66666667, 878.66666667],
                           [8.66666667, 47.66666667, 101., 280., 279.66666667],
                           [17., 99.66666667, 408.33333333, 458.66666667, 1093.],
                           [7.66666667, 48.33333333, 171., 209.66666667, 466.]])

    return exp_values


def test_generate_reps(counts_df):
    """Test generate_reps function"""
    # number of replicates to simulate
    n_reps = 4

    repd_df = generate_reps(counts_df, n_reps=n_reps)
    # expected pseudo samples output
    pseudo_samples = [sample + '_rep_' + str(i + 1) for sample in counts_df['sample'].unique() for i in range(n_reps)]

    assert all([x in repd_df['sample_replicates'].unique() for x in pseudo_samples])


def test_combine(coefficients):
    """Test combine function"""
    conditions = ['cond_1', 'cond_2']
    n_clusters = 5
    n_conds = len(conditions)
    n_sims = 5

    combined_coefs = combine(fit=coefficients, conds=conditions,
                             n_clusters=n_clusters, n_conds=n_conds, n_sims=n_sims)
    # expected combined coefficients
    exp_coefs = np.array([[0.01470994, 0.0489769, 0.16217504, 0.38633384, 0.38780428],
                          [0.0102832, 0.06165028, 0.17238126, 0.28772362, 0.46796164]])

    assert np.allclose(combined_coefs, exp_coefs)


def test_get_mean_sim(counts_list, exp_mean_counts):
    """Test get_mean_sim function"""
    res = get_mean_sim(counts_list)

    assert np.allclose(res.values, exp_mean_counts)
