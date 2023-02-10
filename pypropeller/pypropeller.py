import pandas as pd
import numpy as np
import anndata
from statsmodels.stats.multitest import multipletests

from pypropeller.get_transformed_props import get_transformed_props
from pypropeller.linear_model import *
from pypropeller.ebayes import *
from pypropeller.utils import *


def pypropeller(adata, clusters='cluster', sample='sample', cond='group', transform='logit', robust=True, verbose=True):
    """Test the significance of changes in cell proportions across conditions in single-cell data. The function
    uses empirical bayes to moderate statistical tests to give robust estimation of significance.

    :param anndata.AnnData adata: Anndata object containing single-cell data.
    :param str clusters: Columne in adata.obs where cluster or celltype information are stored, defaults to 'cluster'
    :param str sample: Column in adata.obs where sample informtaion are stored, defaults to 'sample'
    :param str cond: Column in adata.obs where condition informtaion are stored, defaults to 'group'
    :param str transform: Method of normalization of proportions (logit or arcsin), defaults to 'logit'
    :param bool robust: Robust estimation to mitigate the effect of outliers, defaults to True
    :return pandas.DataFrame: Dataframe containing estimated mean proportions for each condition, 
    F-statistics, p-values and adjusted p-values. 
    """
    if isinstance(adata, anndata.AnnData):
        adata = adata.obs
    counts, props, prop_trans = get_transformed_props(adata, sample_col=sample, cluster_col=clusters, transform=transform)
    baseline_props = adata[clusters].value_counts()/adata.shape[0]  # proportions of each cluster in all samples
    #group_coll = pd.crosstab(adata.obs[sample], adata.obs[cond])  # cell counts for each sample across conditions
    #design = group_coll.where(group_coll == 0, 1).astype('int')  # same as group_coll but counts are replaced with 1 and 0
    design = create_design(data=adata, samples=sample, conds=cond)
    
    if design.shape[1] == 2:
        if verbose:
            print("There are 2 conditions. T-Test will be performed...")
        contrasts = [1, -1]  # columns in design matrix to be tested
        out = t_test(props, prop_trans, design, contrasts, robust)
    elif design.shape[1] > 2:
        if verbose:
            print("There are more than 2 conditions. ANOVA will be performed...")
        coef = np.arange(len(design.columns))  # columns of the design matrix corresponding to conditions of interest
        out = anova(props, prop_trans, design, coef, robust)

    print('Done!')

    columns = list(out.columns)
    out['Baseline_props'] = baseline_props.values
    columns.insert(0, 'Baseline_props')
    # rearrange dataframe columns
    out = out[columns]

    return out


def anova(props, prop_trans, design, coef, robust=True, verbose=True):
    """Test the significance of changes in cell proportion across 3 or more conditions using 
    empirical bayes and moderated ANOVA.

    :param pandas.DataFrame props: True cell proportions.
    :param pandas.DataFrame prop_trans: Normalized cell proportions.
    :param pandas.DataFrame design: Design matrix where rows are samples and columns are 
    coefficients of condtions of interest to be estimated.
    :param numpy.ndarray coef: Array specifiying columns of interest in the design matrix.
    :param bool robust: Robust empirical bayes estimation of posterior variances.
    :return pandas.DataFrame: Dataframe containing estimated mean proportions for each condition, 
    F-statistics, p-values and adjusted p-values.
    """
    from statsmodels.tools.tools import add_constant
    if not isinstance(coef, np.ndarray):
        coef = np.array(coef)
    # check if there are less than 3 clusters
    if prop_trans.shape[1] < 3:
        if verbose:
            print("Robust eBayes needs 3 or more clusters! Normal eBayes will be performed")
        robust = False
    X = design.iloc[:,coef]
    N = len(X)  # number of samples
    p = len(X.columns)  # number of conditions
    # fit linear model to each cluster to get coefficients estimates
    fit_prop = lm_fit(X=X, y=props)
    # Change design matrix to intercept format
    design_2 = design.iloc[:,1:]
    design_2 = add_constant(design_2, prepend=True, has_constant='skip')
    # fit fit linear model with all confounding variables
    fit = lm_fit(X=design_2, y=prop_trans)

    # remove intercept from stdev, coefficients and covariance matrix for the ebayes method
    fit['coefficients'] = fit['coefficients'][:,coef[1:]]
    fit['stdev'] = fit['stdev'][:,coef[1:]]
    fit['cov_coef'] = fit['cov_coef'][coef[1:][:, np.newaxis],coef[1:]]
    # get F statistics using eBayes
    fit = ebayes(fit, robust=robust)

    # adjust p_values using benjamin hochberg method
    p_values = fit['F']['F_p_value'].flatten()
    fdr = multipletests(p_values, method='fdr_bh')

    res = {}
    res['Clusters'] = props.columns.to_list()
    for i, cond in enumerate(X.columns):
        res['Mean_props_' + cond] = fit_prop['coefficients'].T[i]
    res['F_statistics'] = fit['F']['stat'].flatten()
    res['p_values'] = p_values
    res['Adjusted_p_values'] = fdr[1]
    cols = list(res.keys())

    return pd.DataFrame(res, columns=cols).set_index('Clusters')


def t_test(props, prop_trans, design, contrasts, robust=True, verbose=True):
    """Test the significance of changes in cell proportion across 2 conditions using 
    empirical bayes and moderated t-test.

    :param pandas.DataFrame props: True cell proportions.
    :param pandas.DataFrame prop_trans: Normalized cell proportions.
    :param pandas.DataFrame design: Design matrix where rows are samples and columns are 
    coefficients of condtions of interest to be estimated.
    :param list contrasts: A list specifiying 2 conditions in the design matrix to be tested; [1, -1].
    :param bool robust: Robust empirical bayes estimation of posterior variances.
    :return pandas.DataFrame: Dataframe containing estimated mean proportions for each condition, 
    F-statistics, p-values and adjusted p-values.
    """
    if prop_trans.shape[1] < 3:
        if verbose:
            print("Robust eBayes needs 3 or more clusters! Normal eBayes will be performed")
        robust = False
    fit = lm_fit(X=design, y=prop_trans)
    fit_cont = contrasts_fit(fit, contrasts)
    fit_cont = ebayes(fit_cont, robust=robust)
    # Get mean cell type proportions and relative risk for output
    # If no confounding variable included in design matrix
    contrasts = np.array(contrasts)
    if len(contrasts) == 2:
        fit_prop = lm_fit(X=design, y=props)
        #z = np.array(list(map(lambda x: x**contrasts ,fit_prop['coefficients']))).T
        z = (fit_prop['coefficients']**contrasts).T
        RR = np.prod(z, axis=0)
    # If confounding variables included in design matrix exclude them
    else:
        design = design.iloc[:, np.where(contrasts!=0)[0]]
        fit_prop = lm_fit(X=design, y=props)
        new_cont = contrasts[contrasts!=0]
        z = (fit_prop['coefficients']**new_cont).T
        RR = np.prod(z, axis=0)

    # adjust p_values using benjamin hochberg method
    p_values = fit_cont['p_value'].flatten()
    fdr = multipletests(p_values, method='fdr_bh')

    # create results dict
    res = {}
    res['Clusters'] = props.columns.to_list()
    for i, cond in enumerate(design.columns):
        res['Mean_props_' + cond] = fit_prop['coefficients'].T[i]

    res['Prop_ratio'] = RR
    res['t_statistics'] = fit_cont['t'].flatten()
    res['p_values'] = p_values
    res['Adjusted_p_values'] = fdr[1]
    cols = list(res.keys())

    return pd.DataFrame(res, columns=cols).set_index('Clusters')
