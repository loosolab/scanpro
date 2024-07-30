from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.stats import binom, nbinom

from scanpro.gaussq2 import gausq2


def del_index(x, indices):
    """Delete rows and columns of a 1 or 2D numpy array based on indices.

    :param numpy.ndarray x: Numpy array.
    :param list indices: List of indices.
    :return numpy.ndarray: Sliced array.
    """
    if x.ndim == 1:
        x = np.delete(x, indices)
    elif x.ndim == 2:
        x = np.delete(x, indices, axis=0)  # delete rows
        x = np.delete(x, indices, axis=1)  # delete columns
    return x


def pmax(x, const):
    """ Function compares each number in a list to a constant and returns the maximum

    :param list x: A list with numbers
    :param float const: A constant to compare to
    :return ndarray: A list with maxima
    """
    return np.array([max([i, const]) for i in x])


def pmin(x, const):
    """ Function to compare each number in a list to a constant and return the minimum

    :param list x: A list with numbers
    :param float const: A constant to compare to
    :return ndarray: A list with minima
    """
    return np.array([min([i, const]) for i in x])


def is_fullrank(x):
    """check if matrix has full rank

    :param np.ndarray x: Matrix.
    :return boolean: True if matrix has full rank, False otherwise.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    e = np.linalg.eig(x.transpose() @ x)[0]
    e[::-1].sort()  # sort descneding
    return e[0] > 1 and np.abs(e[len(e) - 1] / e[0]) > 1e-13


def cov_to_corr(cov):
    """Calculate correlation matrix from covariance matrix

    :param numpy.ndarray cov: A covariance matrix
    :return numpy.ndarray: Corrolation matrix
    """
    # code from https://math.stackexchange.com/questions/186959/correlation-matrix-from-covariance-matrix
    # calculate D^-1 * cov * D^-1 where D^-1 is the square root of inverse of the diagonal of covariance matrix
    if not isinstance(cov, np.ndarray):
        cov = np.array(cov)
    Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
    corr = Dinv @ cov @ Dinv
    return corr


def matvec(M, v):
    """Multiply each row of a matrix by a vector

    :param numpy.ndarray M: A matrix
    :param numpy.ndarray v: A vector to multiply with; length must match number of columns in M!
    :return numpy.ndarray: The matrix resulting from multiplication
    """
    M = np.array(M)
    v = np.array(v[:, np.newaxis])  # reshaping v to be 2D: [1,2] -> [[1],[2]]
    if len(v) != M.shape[1]:
        print(f"Dimensions of M: {M.shape} and v: {v.shape} don't match!")
        return None
    return np.array((M.T * v).T)


def vecmat(v, M):
    """Multiply a vector with a matrix.

    :param numpy.ndarray v: A vector to multiply with; length must match number of columns in M!
    :param numpy.ndarray M: A matrix.
    :return numpy.ndarray: Result of the multiplication
    """
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(M, np.ndarray):
        M = np.array(M)

    v = v[:, np.newaxis]
    return v * M


def gauss_quad_prob(n, dist="uniform", ll=0, u=1, mu=0, sigma=1, alpha=1, beta=1):
    """Given a distribution, calculate nodes and weights of a gaussian quadrature.

    :param int n: Number of nodes.
    :param str dist: A distribution, only uniform is available! defaults to "uniform"
    :param int ll: Lower limit of uniform distribution, defaults to 0
    :param int u: Upper limit of uniform distribution, defaults to 1
    :param int mu: Mean of normal distribution, defaults to 0
    :param int sigma: Standard deviation of normal distribution, defaults to 1
    :param int alpha: Parameter for gamma and beta distribution, defaults to 1
    :param int beta: Parameter for gamma and beta distribution, defaults to 1
    :return numpy.ndarray: 2d list of nodes and weights.
    """
    # initiate variables to store results
    res = np.zeros((2, 1))  # first row is nodes, second is weights
    x = res[0]  # nodes
    w = res[1]  # weights
    n = int(n)
    if n < 0:
        print("Negativ number of nodes is not allowed!")
        return None
    if n == 0:
        res = np.zeros((2, 1))
        return res
    if n == 1:
        dist_dict = {'uniform': lambda: (ll + u) / 2,
                     'beta': lambda: alpha / (alpha + beta),
                     'normal': lambda: mu,
                     'gamma': lambda: alpha * beta}
        x = dist_dict[dist]()
        w = 1
        return res

    i1 = np.arange(1, n)
    if dist == 'uniform':  # skipping other distributions because they are not used
        a = np.zeros(n)
        b = i1 / np.sqrt(4 * (i1**2) - 1)

    b = np.append(b, 0.)
    z = np.zeros(n)
    z[0] = 1
    ierr = 0
    a, b, z = gausq2(n, a, b, z, ierr)
    x = a  # nodes
    w = z**2  # weights
    if dist == 'uniform':  # skipped other dists since we only use uniform!
        x = ll + (u - ll) * (x + 1) / 2
    # save results to 2d list
    res = np.zeros((2, n))
    res[0] = x
    res[1] = w

    return res


def estimate_params_from_counts(x):
    """Estimate paramters for beta distribution from a count matrix

    :param _type_ x: _description_
    :return _type_: _description_
    """
    counts = x
    nc = norm_counts(counts)
    m1 = nc.mean(axis=1)
    m2 = (nc**2).sum(axis=1) / nc.shape[1]
    n = np.mean(nc.sum())
    alpha = (n * m1 - m2) / (n * (m2 / m1 - m1 - 1) + m1)
    beta = ((n - m1) * (n - m2 / m1)) / (n * (m2 / m1 - m1 - 1) + m1)
    disp = 1 / (alpha + beta)
    pi = alpha / (alpha + beta)
    var = n * pi * (1 - pi) * (n * disp + 1) / (1 + disp)
    return [n, alpha, beta, pi, disp, var]


def estimate_beta_params(x):
    """Estimate paramters for beta distribution from proportions matrix

    :param pandas.DataFrame x: A matrix with cells proportions, rows are clusters and columns are samples.
    :return float: Estimated alpha and beta parameters.
    """
    mu = x.mean(axis=1)
    V = x.var(axis=1)
    a = (((1 - mu) / V) - (1 / mu)) * mu**2
    b = (((1 - mu) / V) - (1 / mu)) * mu * (1 - mu)
    return a, b


def norm_counts(x, log=False, prior_count=0.5, lib_size=None):
    """Normalize count matrix to library size.

    :param pandas.DataFrame x: Count matrix.
    :param bool log: _description_, defaults to False
    :param float prior_count: _description_, defaults to 0.5
    :param _type_ lib_size: _description_, defaults to None
    :return _type_: Normalized count matrix.
    """
    counts = x
    if lib_size is None:
        lib_size = np.array(x.sum().to_list())
    M = np.median(lib_size)
    if log:
        prior_counts_scaled = lib_size / np.mean(lib_size) * prior_count
        return np.log2((((counts).T + prior_counts_scaled) / lib_size[:, np.newaxis] * M).T)
    else:
        return ((counts).T / lib_size[:, np.newaxis] * M).T


def simulate_cell_counts(props, n_reps, a, b, n_conds=2, n=20, mu=5000):
    """simulating cell counts using distribution of data.
    - The total numbers of cells for each sample (n_j) are drawn from a negative binomial distribution.
    - The proportions for each cell type in each sample (p_ij) is drawn from beta distribution with parameters a and b.
    - The cell counts for each cluster in each sample are drawn from a binomial distribution with propability p_ij and
    and size (n) = n_ij.

    :param pandas.DataFrame or numpy.ndarray props: True proportions; proportions of each cluster in all samples.
    :param int n_reps: Number of replicates per condition.
    :param list a: estimated alpha paramters for beta distribution.
    :param list b: estimated beta paramters from beta distribution for each cluster in each sample.
    :param int n_conds: number of conditions to be simulated. Depends on a and b.
    :param float n: Parameter (size) for NB distribution, defaults to 20.
    :param float mu: Mean parameter (depth) for NB distribution, defaults to 5000.
    :return pandas.DataFrame: Simulated cell counts.
    """
    # estimate p parameter for negativ binomial from data
    p = n / ((n + mu) if n + mu != 0 else 1)

    # generate total counts for each sample
    total_reps = n_reps * n_conds  # number of replicates multiplied by number of conditions
    num_cells = nbinom.rvs(n, p, size=total_reps)

    # generate sample proportions
    true_p = np.zeros((len(props), total_reps))  # for each condition we will generate n_reps samples
    clusters_names = ['c' + str(i) for i in range(1, len(props) + 1)]

    if isinstance(a, int) or isinstance(a, float):
        a = [a]
    for k in range(len(props)):  # len(props) = props.shape[0]; iterate over clusters
        # draw random proportions from beta distribution for each sample
        if len(a) == 1:
            true_p[k, :] = np.random.beta(a[0], b[k], total_reps)
        else:
            for i in range(0, total_reps, n_reps):
                true_p[k, i:i + n_reps] = np.random.beta(a[k], b[i // n_reps][k], n_reps)

    # generate counts for each cluster in each replicate
    counts_sim = np.zeros((len(true_p), total_reps))
    for i in range(len(props)):
        counts_sim[i, :] = binom.rvs(n=num_cells, p=true_p[i, :], size=total_reps)

    counts_sim = counts_sim.T
    samples_names = ['S' + str(i) for i in range(1, total_reps + 1)]
    conds_names = np.repeat([f"cond_{i}" for i in range(1, n_conds + 1)], total_reps / n_conds)

    counts = pd.DataFrame(counts_sim, columns=pd.Index(clusters_names))
    counts["sample"] = samples_names
    counts["group"] = conds_names

    return counts


def simulate_cell_counts_2(props, n_reps, a, b, n_conds=2, n=20, mu=5000):
    """Simulate cell count matrix with differences in proportions.
    - The total numbers of cells for each sample (n_j) are drwan from a negative binomial distribution.
    - The proportions for each cell type in each sample (p_ij) is drawn from beta distribution with parameters a and b.
    - The cell counts for each cluster in each sample are drawn from a binomial distribution with propability p_ij and
    and size (n) = n_ij.

    :param pandas.DataFrame or numpy.ndarray props: Proportions of each cluster in all samples.
    :param int n_reps: Number of replicates.
    :param list a: estimated alpha paramters for beta distribution.
    :param list b: estimated beta paramters from beta distribution for each cluster in each sample.
    :param float n: Parameter for NB distribution, defaults to 20.
    :param float mu: Mean parameter for NB distribution, defaults to 5000.
    :return pandas.DataFrame: Simulated cell counts.
    """
    # calculate p
    p = n / ((n + mu) if n + mu != 0 else 1)

    # generate total counts for each sample
    total_reps = n_reps * n_conds  # number of reps multiplied by number of conditions

    # draw number of cells for each sample from a negativ binomial distribution
    num_cells = nbinom.rvs(n, p, size=total_reps)

    # generate sample proportions
    true_p = np.zeros((props.shape[0], total_reps))  # for each sample we will generate n_reps replicates
    samples_names = ['S' + str(i) for i in range(1, total_reps + 1)]

    # get clusters names
    try:
        clusters_names = props.index
    except AttributeError:
        clusters_names = [f'c{i + 1}' for i in range(len(props))]

    # draw cluster proportions from a beta distribution
    for k in range(len(props)):  # len(props) = props.shape[0]; iterate over clusters
        for i in range(0, total_reps, n_reps):  # iterate over samples
            # draw random proportions from beta distribution for each sample
            true_p[k, i:i + n_reps] = np.random.beta(a[k], b[i // n_reps][k], n_reps)

    # generate counts for each cluster in each replicate
    counts_sim = np.zeros((len(true_p), total_reps))
    for i in range(len(props)):
        counts_sim[i, :] = binom.rvs(n=num_cells, p=true_p[i, :], size=total_reps)

    counts_sim = counts_sim.T
    counts = pd.DataFrame(counts_sim, columns=clusters_names)
    counts["sample"] = samples_names
    conds_names = np.repeat([f"cond_{i}" for i in range(1, n_conds + 1)], total_reps / n_conds)
    counts["group"] = conds_names

    return counts


def convert_counts_to_df(counts, prop_cols=None, meta_cols=None, n_cells=1, column_name="Cluster"):
    """ Convert a cell count matrix to a dataframe in long format.

    :param pandas.DataFrame counts: cluster*sample cell count matrix.
    :param list prop_cols: List of column names where counts are, defaults to None
    :param list meta_cols: List of column names where additional info are, defaults to None
    :param int n_cells: _description_, defaults to 1
    :param str column_name: Column name in new dataframe where clusters are saved, defaults to "Cluster"
    :return pandas.DataFrame: A dataframe in a long format.
    """

    counts = counts.copy()

    # If not given, try to get prop_cols and meta_cols automatically
    if prop_cols is None:
        dtypes = counts.dtypes.astype(str)
        prop_cols = [col for i, col in enumerate(counts.columns) if "float" in dtypes[i] or "int" in dtypes[i]]

    if meta_cols is None:
        meta_cols = [col for col in counts.columns if col not in prop_cols]

    # Multiply proportions with n_cells
    counts[prop_cols] *= n_cells
    counts[prop_cols] = counts[prop_cols].astype(int)

    # Melt into long format (similar to adata.obs)
    counts_melt = pd.melt(counts, id_vars=meta_cols, value_vars=prop_cols,
                          var_name=column_name, value_name="count")

    # Duplicate rows based on number of cells
    counts_long = counts_melt.loc[counts_melt.index.repeat(counts_melt["count"])].reset_index(drop=True)
    counts_long.drop(columns="count", inplace=True)
    counts_long.index = ["cell_" + str(i) for i in range(1, len(counts_long) + 1)]

    return counts_long
