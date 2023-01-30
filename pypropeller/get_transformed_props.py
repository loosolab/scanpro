import numpy as np
import pandas as pd
from pypropeller.utils import *


def get_transformed_props(adata, sample_col='sample', cluster_col='cluster', transform='logit'):
    """Calculate and normalize proportions using logit or arcsin transformation.

    :param AnnData adata: Anndata object containing single-cell data.
    :param str transform: Method of transformation (logit or arcsin), defaults to 'logit'
    :return pandas.DataFrame: Three dataframes containing counts, proportions and
    transformed proportions, respectively. 
    """
    # get counts for each cluster in each sample
    counts = pd.crosstab(index = adata.obs[sample_col], columns=adata.obs[cluster_col])
    # get sum of cells in sample
    counts['sum'] = counts.sum(axis=1)
    # true proportions for each cluster in each sample -> counts(cluster_in_sample)/sum(counts_in_sample)
    props = counts.iloc[:,:-1].div(counts["sum"], axis=0)
    # transform props
    if transform == 'logit':
        pseudo_counts = counts + 0.5  # adding pseudo count to avoid zeroes
        pseudo_counts['sum'] = pseudo_counts.iloc[:,:-1].sum(axis=1)
        pseudo_props = pseudo_counts.iloc[:,:-1].div(pseudo_counts["sum"], axis=0)
        prop_trans = np.log(pseudo_props/(1-pseudo_props))
    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(props))
    
    return counts.iloc[:,:-1], props, prop_trans


def get_transformed_props_df(df, sample_col='sample', cluster_col='cluster', transform='logit'):
    counts = pd.crosstab(index = df[sample_col], columns=df[cluster_col])
    # get sum of cells in sample
    counts['sum'] = (counts).sum(axis=1)
    # true proportions for each cluster in each sample -> counts(cluster_in_sample)/sum(counts_in_sample)
    props = counts.iloc[:,:-1].div(counts["sum"], axis=0)
    # transform props
    if transform == 'logit':
        pseudo_counts = counts + 0.5  # adding pseudo count to avoid zeroes
        pseudo_counts['sum'] = pseudo_counts.iloc[:,:-1].sum(axis=1)
        pseudo_props = pseudo_counts.iloc[:,:-1].div(pseudo_counts["sum"], axis=0)
        prop_trans = np.log(pseudo_props/(1-pseudo_props))
    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(props))
    
    return counts.iloc[:,:-1], props, prop_trans


def get_tranformed_props_counts(x, transform='logit', replicated=False):
    counts = x.copy()
    counts['sum'] = counts.sum(axis=1)
    # true proportions for each cluster in each sample -> counts(cluster_in_sample)/sum(counts_in_sample)
    props = counts.iloc[:,:-1].div(counts["sum"], axis=0)
    # transform props
    if transform == 'logit':
        if replicated:
            counts = norm_counts(counts)
        pseudo_counts = counts + 0.5  # adding pseudo count to avoid zeroes
        pseudo_counts['sum'] = pseudo_counts.iloc[:,:-1].sum(axis=1)
        pseudo_props = pseudo_counts.iloc[:,:-1].div(pseudo_counts["sum"], axis=0)
        prop_trans = np.log(pseudo_props/(1-pseudo_props))
    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(props))
    
    return props, prop_trans