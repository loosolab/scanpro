import warnings
# import anndata
import scipy
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=anndata.ImplicitModificationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def generate_reps(data, n_reps=2, sample_col='sample', min_rep_pct=0.1, dist='norm'):
    """Generate replicates by splitting original samples using bootstrapping.

    :param anndata.AnnData or pandas.DataFrame data: Dataframe or adata.obs whith single cell info.
    :param int n_reps: Number of replicates to generate, defaults to 2.
    :param str sample_col: Column where samples are stored, defaults to 'sample'.
    :param float min_rep_pct: Avoid choosing very small/large number of cells for each replicate, defaults to 0.1
    :param str dist: Distribution to use for probabilities of drawing number of cells for replicates, defaults to 'norm'.
    :param float std: Standard deviation of number of cells in samples, defaults to 1000.
    :return pandas.DataFrame: List of replicates as dataframes.
    """
    # counts, _, _ = get_transformed_props(data, sample_col=samples, cluster_col='celltype', transform='logit')
    # s = np.sum(counts.T.values, axis=0)
    # n_binom, p_binom = fit_nbinom(s)
    # print(n_binom, p_binom)

    if type(data).__name__ == "AnnData":
        data = data.obs

    samples_list = data[sample_col].unique()
    samples_datas = {}
    for sample in samples_list:
        samples_datas[sample] = data[data[sample_col] == sample]  # subset data for each sample
    indices = {}
    for sample in samples_list:
        indices[sample] = np.arange(samples_datas[sample].shape[0])  # get sequence of indices for each sample [0:n_cells_in_sample]
    n_min = [min([len(indices[sample]) for sample in samples_list])][0]
    reps = []
    for sample in samples_list:  # loop over samples
        # choose n_min cells randomly
        reduce = np.random.choice(indices[sample], n_min, replace=False)
        samples_datas[sample] = samples_datas[sample].iloc[reduce, :]
        n = n_min
        cells = list(samples_datas[sample].index)  # get all cells (after reduction) in data
        # get proportions of cell clusters for each cell
        # -> calculated beforehand and added to data.obs; props=in each sample, props_2=in all samples
        cell_probs = np.array(samples_datas[sample]['props'])
        cell_probs += 0.01  # add pseudo counts to avoid small clusters not getting chosen
        cell_probs /= cell_probs.sum()  # normalize to get sum=1
        for i in range(n_reps):
            if dist == 'norm':
                # TODO: change size parameter to be not hardcoded
                rv = scipy.stats.norm(samples_datas[sample].shape[0] / n_reps, 1039.5)  # 792.5838185170071 -> standard deviation from original counts
                x = np.arange(n)
                if min_rep_pct:
                    x = scipy.stats.trimboth(x, min_rep_pct)  # since we don't want samples to have too small or large counts
                    # proportions[:int(min_rep_pct*n)] = 0
                probabilities = rv.pdf(x)  # generate probabilities using normal distribution
                probabilities = probabilities / probabilities.sum()  # normalize probabilities to get sum=1

            n_rep = np.random.choice(x, p=probabilities)  # number of cells for replicate
            rep_cells = np.random.choice(cells, n_rep, p=cell_probs, replace=False)  # choose n_rep cells
            rep = samples_datas[sample].loc[rep_cells]
            rep.loc[:, sample_col] = [sample + '_rep_' + str(i + 1)] * rep.shape[0]
            reps.append(rep)
            n -= n_rep  # substract number of cells of replicate from total number of cells

            not_chosen_cells = np.where(np.isin(cells, rep_cells, invert=True))[0]
            cells = [cells[i] for i in not_chosen_cells]  # remove chosen cells for next replicate
            cell_probs = cell_probs[not_chosen_cells]  # remove probabilities of chosen cells
            cell_probs /= cell_probs.sum()  # normalize again to get sum=1

    rep_data = pd.concat(reps, join='outer')

    return rep_data


def combine(fit,
            n_sims,
            n_conds,
            conds,
            n_clusters,
            ):
    """Function to combine coefficients estimates from multiple runs of pypropeller,
    following rubin's rule.
    Code adapted from https://github.com/MIDASverse/MIDASpy

    :param dict fit: Dictionary containing beta coefficients estimate for each condition.
    Keys are names of conditions and values are matrices of coefficients for all clusters for each run.
    :param int n_sims: Number of runs of pypropeller. Number must match the number of rows of matrices in fit.
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
