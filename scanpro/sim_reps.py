import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def generate_reps(data, n_reps=8, sample_col='sample', covariates=None):
    """Generate replicates by splitting original samples using bootstrapping.

    :param anndata.AnnData or pandas.DataFrame data: Dataframe or adata.obs whith single cell info.
    :param int n_reps: Number of replicates to generate, defaults to 8.
    :param str sample_col: Column where samples are stored, defaults to 'sample'.
    :return pandas.DataFrame: List of replicates as dataframes.
    """
    # check type of data
    if type(data).__name__ == "AnnData":
        data = data.obs

    samples_list = data[sample_col].unique()

    # subset data for each sample
    samples_datas = {}
    for sample in samples_list:
        # subset data for each sample
        samples_datas[sample] = data[data[sample_col] == sample]

    # name of column to store replicates
    replicate_col = sample_col + "_replicates"

    indices = {}
    for sample in samples_list:
        # get sequence of indices for each sample [0:n_cells_in_sample]
        indices[sample] = np.arange(samples_datas[sample].shape[0])

    # get minimum number of cells in all samples
    n_min = [min([len(indices[sample]) for sample in samples_list])][0]

    reps = []
    for sample in samples_list:  # loop over samples
        # choose n_min cells randomly
        reduce = np.random.choice(indices[sample], n_min, replace=False)  # unique indices of cells
        samples_datas[sample] = samples_datas[sample].iloc[reduce, :]
        n = n_min  # number of cells in a sample before subtracting
        cells_indices = np.arange(n)  # all cells in a sample

        for i in range(n_reps):
            x = range(n)
            n_rep = np.random.choice(x)  # number of cells for replicate
            rep_cells = np.random.choice(cells_indices, n_rep, replace=False)  # choose n_rep cells
            rep = samples_datas[sample].iloc[rep_cells, :].copy()  # get only chosen cells as a dataframe
            rep.loc[:, replicate_col] = [sample + '_rep_' + str(i + 1)] * rep.shape[0]  # add sample name as column
            # add covariate column to avoid one replicate having multiple covariate values
            if covariates:
                if len(rep) > 0:
                    for cov in covariates:
                        x = range(len(rep[cov].unique()))
                        j = np.random.choice(x)
                        rep[cov] = [rep[cov].value_counts().index[j]] * rep.shape[0]
            reps.append(rep)

            n -= n_rep  # substract number of cells of replicate from total number of cells

            # get indices of cells that where not chosen
            not_chosen_cells = np.where(np.isin(cells_indices, rep_cells, invert=True))[0]
            # remove chosen cells for next replicate
            cells_indices = [cells_indices[i] for i in not_chosen_cells]

    # join replicates
    rep_data = pd.concat(reps, join='outer')

    return rep_data


def combine(fit,
            n_sims,
            n_conds,
            conds,
            n_clusters,
            ):
    """ Function to combine coefficients estimates from multiple runs of scanpro,
    following rubin's rule.
    Code adapted from https://github.com/MIDASverse/MIDASpy

    :param dict fit: Dictionary containing beta coefficients estimate for each condition. Keys are names of conditions and values are matrices of coefficients for all clusters for each run.
    :param int n_sims: Number of runs of scanpro. Number must match the number of rows of matrices in fit.
    :param int n_conds: Number of conditions. Must match the number of keys in fit.
    :param list conds: List of names of conditions.
    :param int n_clusters: Number of clusters. Must match number of columns in fit's matrices.
    :return numpy.ndarray: Combined estimates of coefficients.
    """
    mods_est = np.zeros((n_conds, n_clusters))
    m = n_sims

    for i, condition in enumerate(conds):
        Q_bar = np.multiply((1 / m), np.sum(np.array(fit[condition]), 0))
        mods_est[i] = Q_bar

    return mods_est


def get_mean_sim(df_list):
    """Calculate the mean of each index in multiple dataframes.

    :param list df_list: List of pandas dataframes to calculate mean from.
    :return pandas.DataFrame: A dataframe with means.
    """
    df_concat = pd.concat(df_list)
    df_groupby = df_concat.groupby(df_concat.index)
    df_mean = df_groupby.mean()

    return df_mean
