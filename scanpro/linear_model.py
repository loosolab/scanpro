import numpy as np
import pandas as pd
import statsmodels.api as sm
from scanpro.utils import vecmat, del_index, cov_to_corr
from patsy import dmatrix
import re


def lm_fit(X, y):
    """ A function to fit a linear model to each cluster and calculate variance and standard deviation

    :param DataFrame X: A design matrix, rows are samples and columns are condtitions.
    :param DataFrame y: A matrix of clusters propotions, rows are samples and columns are clusters. The output of get_transformed_props.
    :return dict: A dictionary of results from each fit.
    """
    n_clusters = len(y.columns)  # number of clusters
    n_cond = len(X.columns)  # number of conditions
    fit = {}

    # loop over clusters and fit linear model for each cluster seperately
    for i, cluster in enumerate(y.columns):
        M = y.iloc[:, i]  # get proportions for only one cluster
        obs = np.isfinite(M)
        if sum(obs) > 0:
            X_cluster = X[obs]
            M = M[obs]
            model = sm.OLS(M, X_cluster)  # design.iloc[:,coef] is passed to the fit function as design
            out = model.fit()
            fit[cluster] = out

    # initialize lists for results
    coefficients = np.zeros((n_clusters, n_cond))  # list for beta coefficients
    sigma = np.zeros(n_clusters)  # list for variance
    stdev = np.zeros((n_clusters, n_cond))  # list for standard deviations
    df_residual = np.zeros(n_clusters)  # list for residual degrees of freedom
    ssr = np.zeros(n_clusters)  # list for sum of squared residuals
    results = {}

    i = 0  # index for loop
    for cluster, fit in fit.items():

        coefficients[i] = fit.params  # beta
        s = np.sqrt(fit.ssr / (fit.df_resid))  # sigma calculated as sum of squared residuals / residual degree of freedom
        sigma[i] = s
        stdev[i] = fit.params / fit.tvalues / s  # standard deviation
        df_residual[i] = fit.df_resid  # residual degrees of freedom
        ssr[i] = fit.ssr  # sum of squared residuals
        i += 1

    # append results to dictionary
    results['coefficients'] = coefficients
    results['sigma'] = sigma
    results['stdev'] = stdev
    results['df_residual'] = df_residual
    results['ssr'] = ssr
    results['design'] = X

    # calculate the covariance using QR decomposition
    m = np.linalg.qr(X, mode='r')
    cov_coef = np.linalg.inv((m.T @ m))
    results['cov_coef'] = cov_coef

    return results


def contrasts_fit(fit_prop, contrasts=None, coefficients=None):
    """Given a set of contrasts, estimate coefficients and standard error of given linear model fit.

    :param dict fit: Dictionary of fitted linear models resulting from lm_fit.
    :param list contrasts: Array of contrasts, defaults to None
    :param list coefficients: List of coefficints, defaults to None
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return dict: Dictionary of estimated values for each fit given array of contrasts.
    """
    fit = fit_prop.copy()

    # check for correct fitting values
    if contrasts is None and coefficients is None:
        raise ValueError("Specify either contrasts or coefficients!")
    if 'coefficients' not in fit.keys():
        raise ValueError("Fit must contain coefficients!")
    if 'stdev' not in fit.keys():
        raise ValueError("Fit must contain standard deviations!")

    # Remove test statistics in case eBayes() has previously been run on the fit object
    for key in ['t', 'p_value', 'lods', 'F']:
        if key in fit.keys():
            fit[key] = None

    n_coef = fit['coefficients'].shape[1]
    if any(np.isnan(contrasts)):
        raise ValueError("Contrasts should not contain NaN values!")

    contrasts = np.array(contrasts)
    contrasts = contrasts.reshape(len(contrasts), 1)
    if contrasts.shape[0] != n_coef:
        raise ValueError("Number of contrasts doesnt match number of coefficients!")
    fit['contrasts'] = contrasts
    # Correlation matrix of estimable coefficients
    # Test whether design was orthogonal
    cor_matrix = cov_to_corr(fit['cov_coef'])
    if len(np.array(cor_matrix.ravel())) < 2:
        orthog = True
    else:
        orthog = sum(abs(cor_matrix[np.tril_indices(cor_matrix.shape[0], -1)])) < 1e-12

    # If design matrix was singular, reduce to estimable coefficients
    r = cor_matrix.shape[0]
    if r < n_coef:  # not needed, since r == n_coef
        pass

    contrasts_all_zero = np.where(np.sum(abs(contrasts), axis=1) == 0)[0]  # which function -> indices where condition is met
    if contrasts_all_zero.any():
        # delete rows and columns where contrast = 0
        contrasts = np.delete(contrasts, contrasts_all_zero, axis=0)
        fit['coefficients'] = np.delete(fit['coefficients'], contrasts_all_zero, axis=1)  # delete columns where contrasts = 0
        fit['stdev'] = np.delete(fit['stdev'], contrasts_all_zero, axis=1)
        fit['cov_coef'] = del_index(fit['cov_coef'], contrasts_all_zero)
        cor_matrix = del_index(cor_matrix, contrasts_all_zero)
        n_coef = fit['coefficients'].shape[1]

    # Replace NA coefficients with large (but finite) standard deviations
    # to allow zero contrast entries to clobber NA coefficients.
    na_coef = np.isnan(fit['coefficients']).any()
    if na_coef:
        i = np.isnan(fit['coefficients'])
        fit['coefficients'][i] = 0
        fit['stdev'][i] = 1e30

    # New coefficients
    fit['coefficients'] = fit['coefficients'] @ contrasts

    # Test whether design was orthogonal
    if len(np.array(cor_matrix).ravel()) < 2:  # convert numpy matrix to array before flattining
        orthog = True
    else:
        orthog = np.all(abs(cor_matrix[np.tril_indices(cor_matrix.shape[0], -1)]) < 1e-14)

    R = np.linalg.cholesky(fit['cov_coef']).T
    fit['cov_coef'] = (R @ contrasts).T @ (R @ contrasts)
    # New standard deviations
    if orthog:
        fit['stdev'] = np.sqrt(fit['stdev']**2 @ contrasts**2)
    else:
        R = np.linalg.cholesky(cor_matrix).T  # cor_matrix is np.matrix
        n_clusters = fit['stdev'].shape[0]
        n_cont = contrasts.shape[1]
        U = np.ones((n_clusters, n_cont))
        o = np.ones((1, n_coef))
        for i in range(n_clusters):
            RUC = R @ vecmat(fit['stdev'][i], contrasts)
            U[i] = np.sqrt(o @ RUC**2)
        fit['stdev'] = U

    # Replace NAs if necessary
    if na_coef:
        i = fit['stdev'] > 1e20
        fit['coefficients'][i] = np.nan
        fit['stdev'][i] = np.nan

    return fit


def create_design(data, sample_col, conds_col, covariates=None):
    """Create design matrix where rows=samples and columns=conditions, to use for fitting linear models to clusters.

    :param [pd.DataFrame, anndata.Anndata] data: A dataframe where sample_col, conds_col and covariates are columns
    :param [list, numpy.array or str] samples: If data is provided, samples is the columns name where
    samples are saved, otherwise samples is a list or array of samples.
    :param [list, numpy.array or str] conds: If data is provided, conds is the columns name where
    conditions are saved, otherwise conds is a list or array of conditions corresponding to samples.
    :param [str or dict] covariates: If data is provided, covariates is a string (or list of strings) where
    covariates are saved, otherwise provide a dictionary with keys as covariates names and values are lists
    of covariates corresponding to samples, defaults to None.
    :param anndata.Anndata or pandas.DataFrame data: A dataframe where samples and conditions are columns,
    if data is anndata, samples and conditions must be columns in adata.obs, defaults to None.
    :param bool or list reorder: Reorder columns of data matrix to match the list provided, defaults to False.
    :param bool or list reindex: Reorder rows of data matrix to match the list provided, defaults to False.
    :param bool intercept: If True, an intercept is added as first column in the design matrix, defaults to False
    :param bool before_reindex: If True, covariates are added to design matrix before reordering rows,
    make sure that provided list match the samples, defaults to True.
    :raises TypeError: _description_
    :raises ValueError: _description_
    :return pandas.DataFrame: Design matrix as pandas dataframe.
    """

    # check if data is either anndata or pandas dataframe
    if type(data).__name__ == "AnnData":
        data = data.obs
    elif not isinstance(data, pd.DataFrame):  # if data is not anndata or pandas dataframe
        raise TypeError("Only anndata objects and pandas dataframes are supported!")
    data = data.copy()

    if covariates is None:
        covariates = []

    # Build sample matrix
    cols = [sample_col, conds_col] if sample_col != conds_col else [conds_col]  # prevent duplicated columns
    sample_info = data[cols + covariates].drop_duplicates()
    sample_info.set_index(sample_col, drop=False, inplace=True)

    # build formula
    formula = "~0 + " + " + ".join([conds_col] + covariates)

    # build design matrix
    design = dmatrix(formula, sample_info, return_type='dataframe')

    # Rename columns from "conds_col[cond]" to "cond"
    design.columns = [re.match(r".+\[(T\.){0,1}(.+)\]", col).group(2) for col in design.columns]

    # Reorder columns to be in the same order as the input condition order
    conditions = sample_info[conds_col].unique().tolist()
    sorted_columns = sorted(design.columns, key=lambda x: conditions.index(x) if x in conditions else np.inf)
    design = design[sorted_columns]

    return design
