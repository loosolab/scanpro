import numpy as np
import pandas as pd
from scanpro.utils import norm_counts


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
    props = counts.div(counts.sum(axis=1), axis=0)

    # true proportions for each cluster in each sample -> counts(cluster_in_sample)/sum(counts_in_sample)
    pseudo_counts = counts + 0.01
    # pseudo_counts = counts + 0.5  # to avoid "f(a) and f(b) must have different signs" error
    pseudo_props = pseudo_counts.div(pseudo_counts.sum(axis=1), axis=0)

    # transform props
    if transform == 'logit':
        prop_trans = np.log(pseudo_props / (1 - pseudo_props))

    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(pseudo_props))

    return counts, props, prop_trans


def get_transformed_props_counts(x, transform='logit', sample_col=None, meta_cols=None, normalize=False):
    """Calculate proportions and transformed proportions from a cluster*sample matrix.

    :param pandas.DataFrame x: A count matrix where rows=samples and columns=clusters.
    :param str transform: Method of transformation, defaults to 'logit'
    :param str sample_col: Name of columns where samples are stored. If None, sample_col is index, defaults to None.
    :param list meta_cols: Names of columns where extra sample information are stored, if None, there are no meta columns,
    defaults to None.
    :param bool normalize: If True, count matrix is normalized to the mean of library size, defaults to False
    :return pandas.DataFrame: Two matrices as pandas dataframes; proportions and transformed proportions.
    """
    counts = x.copy()
    if sample_col:
        counts.set_index(sample_col, inplace=True)
    if meta_cols:
        counts = counts[[column for column in counts.columns if column not in meta_cols]]
    # adding pseudo count to avoid zeroes
    pseudo_counts = counts + 0.01
    # calculate cell proportions using real counts
    props = counts.div(counts.sum(axis=1), axis=0)
    # get sum of cells in sample
    pseudo_props = pseudo_counts.div(pseudo_counts.sum(axis=1), axis=0)
    # transform props
    if transform == 'logit':
        if normalize:
            pseudo_counts = norm_counts(pseudo_counts)
        pseudo_props = pseudo_counts.div(pseudo_counts.sum(axis=1), axis=0)
        prop_trans = np.log(pseudo_props / (1 - pseudo_props))
    elif transform == 'arcsin':
        prop_trans = np.arcsin(np.sqrt(pseudo_props))

    return props, prop_trans
