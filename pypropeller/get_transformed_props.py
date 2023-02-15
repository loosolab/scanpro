import numpy as np
import pandas as pd
from pypropeller.utils import norm_counts


def get_transformed_props(data, sample_col='sample', cluster_col='cluster', transform='logit'):
    """Calculate and normalize proportions using logit or arcsin transformation.

    :param [pandas.DataFrame, anndata.AnnData] data: Anndata object or pandas dataframe containing single-cell data.
    :param str transform: Method of transformation (logit or arcsin), defaults to 'logit'
    :return pandas.DataFrame: Three dataframes containing counts, proportions and
    transformed proportions, respectively.
    """

    # check if data is a pandas dataframe or anndata object
    if not type(data).__name__ == "AnnData" and not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be anndata object or a pandas dataframe!")

    if type(data).__name__ == "AnnData":
        data = data.obs

    # get counts for each cluster in each sample
    counts = pd.crosstab(index=data[sample_col], columns=data[cluster_col])
    # get sum of cells in sample
    counts['sum'] = counts.sum(axis=1)
    # true proportions for each cluster in each sample -> counts(cluster_in_sample)/sum(counts_in_sample)
    props = counts.iloc[:, :-1].div(counts["sum"], axis=0)
    # transform props
    if transform == 'logit':
        pseudo_counts = counts + 0.5  # adding pseudo count to avoid zeroes
        pseudo_counts['sum'] = pseudo_counts.iloc[:, :-1].sum(axis=1)
        pseudo_props = pseudo_counts.iloc[:, :-1].div(pseudo_counts["sum"], axis=0)
        prop_trans = np.log(pseudo_props / (1 - pseudo_props))
    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(props))

    return counts.iloc[:, :-1], props, prop_trans


def get_tranformed_props_counts(x, transform='logit', normalize=False):
    """Calculate proportions and transformed proportions from a cluster*sample matrix.

    :param pandas.DataFrame x: A count matrix where rows=samples and columns=clusters.
    :param str transform: Method of transformation, defaults to 'logit'
    :param bool normalize: If True, count matrix is normalized to the mean of library size, defaults to False
    :return pandas.DataFrame: Two matrices as pandas dataframes; proportions and transformed proportions.
    """
    counts = x.copy()
    counts['sum'] = counts.sum(axis=1)
    # true proportions for each cluster in each sample -> counts(cluster_in_sample)/sum(counts_in_sample)
    props = counts.iloc[:, :-1].div(counts["sum"], axis=0)
    # transform props
    if transform == 'logit':
        if normalize:
            counts = norm_counts(counts)
        pseudo_counts = counts + 0.5  # adding pseudo count to avoid zeroes
        pseudo_counts['sum'] = pseudo_counts.iloc[:, :-1].sum(axis=1)
        pseudo_props = pseudo_counts.iloc[:, :-1].div(pseudo_counts["sum"], axis=0)
        prop_trans = np.log(pseudo_props / (1 - pseudo_props))
    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(props))

    return props, prop_trans
