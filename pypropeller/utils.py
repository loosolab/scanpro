from __future__ import print_function

import numpy as np

from pypropeller import gaussq2


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
    gaussq2.gausq2(n, a, b, z, ierr)
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
    """Estimate paramters for beta distribution from acount matrix

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
