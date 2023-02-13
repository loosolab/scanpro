from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.stats import nbinom, binom, t
from scipy.special import gammaln
from scipy.special import psi
from scipy.special import factorial
from scipy.optimize import fmin_l_bfgs_b as optim

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
    return e[0] > 1 and np.abs(e[len(e)-1]/e[0]) > 1e-13


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
    if not isinstance (M, np.ndarray):
        M = np.array(M)
        
    v = v[:, np.newaxis]
    return v * M


def gauss_quad_prob(n, dist="uniform", l=0, u=1, mu=0, sigma=1, alpha=1, beta=1):
    """Given a distribution, calculate nodes and weights of a gaussian quadrature.

    :param int n: Number of nodes.
    :param str dist: A distribution, only uniform is available! defaults to "uniform"
    :param int l: Lower limit of uniform distribution, defaults to 0
    :param int u: Upper limit of uniform distribution, defaults to 1
    :param int mu: Mean of normal distribution, defaults to 0
    :param int sigma: Standard deviation of normal distribution, defaults to 1
    :param int alpha: Parameter for gamma and beta distribution, defaults to 1
    :param int beta: Parameter for gamma and beta distribution, defaults to 1
    :return numpy.ndarray: 2d list of nodes and weights.
    """
    #from pypropeller.gaussq2 import gausq2
    res = np.zeros((2,1))  # first row is nodes, second is weights
    x = res[0]  # nodes
    w = res[1]  # weights
    n = int(n)
    if n < 0:
        print("Negativ number of nodes is not allowed!")
        return None
    if n == 0:
        res = np.zeros((2,1))
        return res
    if n == 1:
        dist_dict = {'uniform': lambda: (l+u)/2, 
                    'beta': lambda: alpha/(alpha+beta),
                    'normal': lambda: mu,
                    'gamma': lambda: alpha*beta}
        x = dist_dict[dist]()
        w = 1
        return res

    i = np.arange(1,n+1)
    i1 = np.arange(1,n)
    if dist=='uniform':  # skipping other distributions because they are not used
        a = np.zeros(n)
        b = i1 / np.sqrt(4*(i1**2)-1)
    
    b = np.append(b, 0.)
    z = np.zeros(n)
    z[0] = 1
    ierr = 0
    gaussq2.gausq2(n, a, b, z, ierr)
    x = a  # nodes
    w = z**2
    if dist == 'uniform':  # skipped other dists since we only use uniform!
        x = l + (u-l)*(x+1)/2
    # save results to 2d list
    res = np.zeros((2,n))
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
    m2 = (nc**2).sum(axis=1)/nc.shape[1]
    n = np.mean(nc.sum())
    alpha = (n*m1-m2)/(n*(m2/m1-m1-1)+m1)
    beta = ((n-m1)*(n-m2/m1))/(n*(m2/m1-m1-1)+m1)
    disp = 1/(alpha+beta)
    pi = alpha/(alpha+beta)
    var = n*pi*(1-pi)*(n*disp+1)/(1+disp)
    return [n, alpha, beta, pi, disp, var]


def estimate_beta_params(x):
    """Estimate paramters for beta distribution from proportions matrix

    :param pandas.DataFrame x: A matrix with cells proportions, rows are clusters and columns are samples.
    :return float: Estimated alpha and beta parameters.
    """
    mu = x.mean(axis=1)
    V = x.var(axis=1)
    a = (((1-mu)/V) - (1/mu))*mu**2
    b = (((1-mu)/V) - (1/mu))*mu*(1-mu)
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
        prior_counts_scaled = lib_size/np.mean(lib_size)*prior_count
        return np.log2((((counts).T+prior_counts_scaled)/lib_size[:,np.newaxis]*M).T)
    else:
        return ((counts).T/lib_size[:,np.newaxis]*M).T


def boot_reps(adata, n_boots = 10, samples='sample'):
    """Generate replicates for each sample using bootstrapping with replacement.

    :param anndata.AnnData adata: Anndata object.
    :param int n_boots: Number of replicates, defaults to 10
    :param str samples: Name of samples column in obs table, defaults to 'sample'
    :return list: list of replicates as anndata objects.
    """
    groups = adata.obs[samples].unique()
    groups_adatas = {}
    for group in groups:
        groups_adatas[group] = adata.obs[adata.obs[samples] == group]  # subset data for each sample
    indices = {}
    for i, group in enumerate(groups_adatas.keys()):
        indices[groups[i]] = np.arange(groups_adatas[group].shape[0])  # get sequence of indices
    reps = []
    for group in groups:
        for i in range(n_boots):
            boot = np.random.choice(indices[group], len(indices[group]))
            rep = groups_adatas[group].iloc[boot,:]
            rep[samples] = [group+'_rep_'+str(i+1)]*rep.shape[0]
            #rep.obs_names_make_unique()
            reps.append(rep)

    return reps


def simulate_cell_counts(counts, props, n_reps, a, b, n=None, mu=None):
    """Generate replicates by simulating cell counts using distribution of data.
    - The total numbers of cells for each sample (n_j) are drwan from a negative binomial distribution.
    - The proportions for each cell type in each sample (p_ij) is drawn from beta distribution with parameters a and b.
    - The cell counts for each cluster in each sample are drawn from a binomial distribution with propability p_ij and
    and size (n) = n_ij.

    :param _type_ counts: Count matrix, rows=clusters, columns=samples.
    :param _type_ props: True proportions; proportions of each cluster in all samples.
    :param int n_reps: Number of replicates.
    :param list a: estimated alpha paramters for beta distribution.
    :param list b: estimated beta paramters from beta distribution for each cluster in each sample.
    :param float n: Parameter for NB distribution, defaults to None
    :param float mu: Mean parameter for NB distribution, defaults to None
    :return pandas.DataFrame: Simulated cell counts.
    """
    # estimate negativ binomial parameters from data
    # get sum of cell counts for each sample
    s = np.sum(counts.values, axis=0)
    #X = np.ones_like(s)
    # fit negative binomial model
    #nb = sm.NegativeBinomial(s, X, full_output=0)
    #res = nb.fit(start_params=[1,1])
    # calculate n and p
    #mu = np.exp(res.params[0])  # or np.mean(s)
    #mu = np.mean(s)
    #var = np.var(s)
    #var = mu + res.params[1]*(mu**2)
    #n = mu**2 / (var - mu) if var > mu else 10
    #n = 20
    #mu = 5000
    #p = n / ((n+mu) if n+mu != 0 else 1)
    if n is None:
        n, p = fit_nbinom(s)
    elif n and mu is None:
        mu = np.mean(s)
        p = n / ((n+mu) if n + mu != 0 else 1)
    #p = mu / var
    #p = 1/((1+mu)*res.params[1])
    #n = mu*p/(1-p)  # dispersion
    # generate total counts for each sample
    total_reps = n_reps*counts.shape[1]  # number of reps multiplied by number of samples
    num_cells = nbinom.rvs(n, p, size=total_reps)
    # generate sample proportions
    true_p = np.zeros((props.shape[0], total_reps))  # for each sample we will generate n_reps replicates
    reps_names = [name + '_rep' + str(i) for name in counts.T.index for i in range(1,n_reps+1)]
    for k in range(len(props)):  # len(props) = props.shape[0]; iterate over clusters
        for i in range(0,total_reps,n_reps):  # iterate over samples
            true_p[k,i:i+n_reps] = np.random.beta(a[k], b[i//n_reps][k], n_reps)  # draw random proportions from beta distribution for each sample

    # generate counts for each cluster in each replicate
    counts_sim = np.zeros((len(true_p), total_reps))
    for i in range(len(props)):
        counts_sim[i,:] = binom.rvs(n=num_cells, p=true_p[i,:], size=total_reps)

    counts_sim = counts_sim.T
    counts_t = counts.T  # counts should have rows=samples and columns=clusters for further analysis

    return pd.DataFrame(counts_sim, index=pd.Index(reps_names, name=counts_t.index.name), columns=counts_t.columns)


def fit_nbinom(X, initial_params=None):
    """Function to fit negativ binomial distribution. See https://github.com/gokceneraslan/fit_nbinom

    :param numpy.ndarray X: numpy array representing the data.
    :param numpy.ndarray initial_params: Initial values for n and p, defaults to None.
    :return float: Size (n) and propabilities (p) parameters.
    """
    infinitesimal = np.finfo(np.float).eps

    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        #MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = np.sum(gammaln(X + r)) \
            - np.sum(np.log(factorial(X))) \
            - N*(gammaln(r)) \
            + N*r*np.log(p) \
            + np.sum(X*np.log(1-(p if p < 1 else 1-infinitesimal)))

        return -result

    def log_likelihood_deriv(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        pderiv = (N*r)/p - np.sum(X)/(1-(p if p < 1 else 1-infinitesimal))
        rderiv = np.sum(psi(X + r)) \
            - N*psi(r) \
            + N*np.log(p)

        return np.array([-rderiv, -pderiv])

    if initial_params is None:
        # reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m**2)/(v-m) if v > m else 10

        # convert mu/size parameterization to prob/size
        p0 = size / ((size+m) if size+m != 0 else 1)
        r0 = size
        initial_params = np.array([r0, p0])

    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     #fprime=log_likelihood_deriv,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    params = optimres[0]
    n = params[0]
    p = params[1]

    return n, p
