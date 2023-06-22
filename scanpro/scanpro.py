import time
import warnings
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant

from scanpro.get_transformed_props import get_transformed_props
from scanpro.linear_model import lm_fit, contrasts_fit, create_design
from scanpro import ebayes
from scanpro.sim_reps import generate_reps, combine, get_mean_sim
from scanpro.result import ScanproResult


def scanpro(data, clusters_col, conds_col, samples_col=None,
            transform='logit', conditions=None, robust=True, n_sims=100, n_reps=8, verbose=True):
    """Wrapper function for scanpro. The data must have replicates,
    since propeller requires replicated data to run. If the data doesn't have
    replicates, the function {sim_scanpro} will generate artificial replicates
    using bootstrapping and run propeller multiple times. The values are then pooled
    to get robust estimation of p values.

    :param anndata.AnnData or pandas.DataFrame data: Single cell data with columns containing sample,
    condition and cluster/celltype information.
    :param str clusters_col: Name of column in date or data.obs where cluster/celltype information are stored.
    :param str conds_col: Column in data or data.obs where condition information are stored.
    :param str samples_col: Column in data or data.obs where sample information are stored, if None,
    dataset is assumed to be not replicated and conds_col will be set as samples_col, defaults to None.
    :param str transform: Method of transformation of proportions, defaults to 'logit'.
    :param str conditions: List of condtitions of interest to compare, defaults to None.
    :param bool robust: Robust ebayes estimation to mitigate the effect of outliers, defaults to True.
    :param int n_sims: Number of simulations to perform if data does not have replicates, defaults to 100.
    :param int n_reps: Number of replicates to simulate if data does not have replicates, defaults to 8.
    :param bool verbose: defaults to True.
    :raises ValueError: Data must have at least two conditions!
    :return scanpro: A scanpro object containing estimated mean proportions for each cluster and p-values.
    """
    if type(data).__name__ == "AnnData":
        data = data.obs
    data = data.copy()  # make sure original data is not modified

    # check if samples_col and conds_col are in data
    columns = [clusters_col, conds_col]
    # add samples_col if given
    if samples_col is not None:
        columns.append(samples_col)

    columns_not_in_data = np.isin(columns, data.columns, invert=True)
    check_columns = any(columns_not_in_data)
    if check_columns:
        s1 = "The following columns could not be found in data: "
        s2 = ', '.join([columns[i] for i in np.where(columns_not_in_data)[0]])
        raise ValueError(s1 + s2)

    # check conditions
    if conditions is not None:
        # check if conditions are in a list
        if not isinstance(conditions, list) and not isinstance(conditions, np.ndarray):
            raise ValueError("Please provide names of conditions of interest as a list!")
        # check if conditions are in data
        not_in_data = np.isin(conditions, data[conds_col].unique(), invert=True)
        check = any(not_in_data)
        if check:
            s1 = "The following conditions could not be found in data: "
            s2 = ', '.join([conditions[i] for i in np.where(not_in_data)[0]])
            raise ValueError(s1 + s2)

    else:
        # if no conditions are specified, get all conditions
        conditions = data[conds_col].unique()

    # check if there are 2 conditions or more
    if len(conditions) < 2:
        raise ValueError("There has to be at least two conditions to compare!")

    # if samples_col is None, data is not replicated
    if samples_col is None:
        repd = False
        partially_repd = False

    # otherwise, assume data is replicated
    else:
        repd = True
        partially_repd = False

        # check if at least one condition doesnt have replicates
        no_reps_list = []
        for condition in conditions:
            # get samples for condition
            samples_list = data[data[conds_col] == condition][samples_col].unique()
            if len(samples_list) == 1:
                no_reps_list.append(condition)

        # at least one condition doesn't have replicates
        if len(no_reps_list) > 0:
            # if all conditions don't have replicates, set repd to False
            if len(no_reps_list) == len(conditions):
                repd = False
            # otherwise, data is partially replicated
            else:
                partially_repd = True

    # check if there are no replicates
    if not repd:
        if verbose:
            print("Your data doesn't have replicates! Artificial replicates will be simulated to run scanpro.")
            if transform != "arcsin":
                print("Consider setting transform='arcsin', as this produces more accurate results for simulated data.")
            print("Simulation may take some minutes...")

        # set transform to arcsin, since it produces more accurate results for simulations
        out = sim_scanpro(data, n_reps=n_reps, n_sims=n_sims, clusters_col=clusters_col,
                          samples_col=samples_col, conds_col=conds_col, transform=transform,
                          conditions=conditions, robust=robust, verbose=verbose)

    # if at least on condition doesn't have replicate, merge samples and bootstrap
    elif partially_repd:
        s1 = "The following conditions don't have replicates: "
        s2 = ", ".join(no_reps_list) + '\n'
        s3 = "Both normal scanpro and sim_scanpro will be performed."
        if verbose:
            print(s1 + s2 + s3)

        # run scanpro normally
        if verbose:
            print("Running scanpro with original replicates...")
        out = run_scanpro(data, clusters_col, samples_col, conds_col, transform,
                          conditions, robust, verbose)

        # run simulations
        if verbose:
            print("Running scanpro with simulated replicates...")

        # add conditions as merged_samples column
        merged_samples_col = 'merged_samples'
        data[merged_samples_col] = data[conds_col]

        # set transform to arcsin, since it produces more accurate results for simulations
        transform = 'arcsin'
        out_sim = sim_scanpro(data, n_reps=n_reps, n_sims=n_sims, clusters_col=clusters_col,
                              samples_col=merged_samples_col, conds_col=conds_col, transform=transform,
                              conditions=conditions, robust=robust, verbose=verbose)

        print("To access results for original replicates, run <out.results>, and <out.sim_results> for simulated results")

    # if all conditions have replicates, run scanpro normally
    else:
        out = run_scanpro(data, clusters_col, samples_col, conds_col, transform,
                          conditions, robust, verbose)

    # add baseline proportions as first column
    baseline_props = data[clusters_col].value_counts() / data.shape[0]  # proportions of each cluster in all samples
    out.results['baseline_props'] = baseline_props.values
    columns = list(out.results.columns)
    columns = ["baseline_props"] + [column for column in columns if column != 'baseline_props']  # put baseline_props first

    # rearrange dataframe columns
    out.results = out.results[columns]

    # add conditions to object
    out.conditions = conditions

    # if data is not replicated, add results also as sim_results for plotting
    if not repd:
        out.sim_results = out.results

    # add simulated results for partially replicated data
    if partially_repd:
        out.sim_results = out_sim.results
        out.sim_design = out_sim.sim_design
        out.sim_counts = out_sim.sim_counts
        out.sim_props = out_sim.sim_props
        out.sim_prop_trans = out_sim.sim_prop_trans

    return out


def run_scanpro(data, clusters, samples, conds, transform='logit',
                conditions=None, robust=True, verbose=True):
    """Test the significance of changes in cell proportions across conditions in single-cell data. The function
    uses empirical bayes to moderate statistical tests to give robust estimation of significance.

    :param anndata.AnnData or pandas.DataFrame adata: Anndata object containing single-cell data.
    :param str clusters: Column in adata.obs where cluster or celltype information are stored.
    :param str samples: Column in adata.obs where sample information are stored.
    :param str conds: Column in adata.obs where condition information are stored.
    :param str transform: Method of normalization of proportions (logit or arcsin), defaults to 'logit'
    :param str conditions: List of condtitions of interest to compare, defaults to None.
    :param bool robust: Robust ebayes estimation to mitigate the effect of outliers, defaults to True
    :return pandas.DataFrame: Dataframe containing estimated mean proportions for each cluster and p-values.
    """
    # check data type
    if type(data).__name__ == "AnnData":
        data = data.obs

    # calculate proportions and transformed proportions
    counts, props, prop_trans = get_transformed_props(data, sample_col=samples, cluster_col=clusters, transform=transform)

    # create design matrix
    design = create_design(data=data, samples=samples, conds=conds, reindex=props.index)

    contrasts = None
    coef = None

    if conditions is not None:
        # get indices of giving conditions in design matrix
        cond_indices = design.columns.get_indexer(conditions)
        if len(cond_indices) == 2:  # if only two conditions, make contrasts list
            contrasts = np.zeros(len(design.columns))
            contrasts[cond_indices[0]] = 1
            contrasts[cond_indices[1]] = -1
        else:
            coef = np.sort(cond_indices)
    # if no specific conditions are provided, consider all conditions
    else:
        # check number of conditions
        if design.shape[1] == 2:
            contrasts = [1, -1]  # columns in design matrix to be tested
        elif design.shape[1] > 2:
            coef = np.arange(len(design.columns))  # columns of the design matrix corresponding to conditions of interest

    if contrasts is not None:
        if verbose:
            print("There are 2 conditions. T-Test will be performed...")
        out = t_test(props, prop_trans, design, contrasts, robust)
    else:
        if verbose:
            print("There are more than 2 conditions. ANOVA will be performed...")
        out = anova(props, prop_trans, design, coef, robust)

    if verbose:
        print('Done!')

    # create scanpro object
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Pandas doesn't allow columns to be created via a new attribute name*")

        output_obj = ScanproResult()
        output_obj.results = out
        output_obj.counts = counts
        output_obj.props = props
        output_obj.prop_trans = prop_trans
        output_obj.design = design

    return output_obj


def anova(props, prop_trans, design, coef, robust=True, verbose=True):
    """ Test the significance of changes in cell proportion across 3 or more conditions using
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
    # check if coef is a numpy array
    if not isinstance(coef, np.ndarray):
        coef = np.array(coef)

    # check if there are less than 3 clusters
    if prop_trans.shape[1] < 3:
        if verbose:
            print("Robust eBayes needs 3 or more clusters! Normal eBayes will be performed")
        robust = False

    X = design.iloc[:, coef]
    # fit linear model to each cluster to get coefficients estimates
    fit_prop = lm_fit(X=X, y=props)

    # Change design matrix to intercept format
    design_2 = design.iloc[:, 1:]
    design_2 = add_constant(design_2, prepend=True, has_constant='skip')
    # fit fit linear model with all confounding variables
    fit = lm_fit(X=design_2, y=prop_trans)

    # remove intercept from stdev, coefficients and covariance matrix for the ebayes method
    fit['coefficients'] = fit['coefficients'][:, coef[1:]]
    fit['stdev'] = fit['stdev'][:, coef[1:]]
    fit['cov_coef'] = fit['cov_coef'][coef[1:][:, np.newaxis], coef[1:]]

    # get F statistics using eBayes
    fit = ebayes.ebayes(fit, robust=robust)

    # adjust p_values using benjamin hochberg method
    p_values = fit['F']['F_p_value'].flatten()
    fdr = multipletests(p_values, method='fdr_bh')

    # save results to dictionary
    res = {}
    res['clusters'] = props.columns.to_list()
    for i, cond in enumerate(X.columns):
        res['mean_props_' + cond] = fit_prop['coefficients'].T[i]
    res['f_statistics'] = fit['F']['stat'].flatten()
    res['p_values'] = p_values
    res['adjusted_p_values'] = fdr[1]
    cols = list(res.keys())

    return pd.DataFrame(res, columns=cols).set_index('clusters')


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
    # check if there are less than 3 clusters
    if prop_trans.shape[1] < 3:
        if verbose:
            print("Robust eBayes needs 3 or more clusters! Normal eBayes will be performed")
        robust = False

    # fit linear model to each cluster to get coefficients estimates
    fit = lm_fit(X=design, y=prop_trans)
    fit_cont = contrasts_fit(fit, contrasts)

    # run empirical bayes
    fit_cont = ebayes.ebayes(fit_cont, robust=robust)

    # Get mean cell type proportions and relative risk for output
    # If no confounding variable included in design matrix
    contrasts = np.array(contrasts)
    if len(contrasts) == 2:
        fit_prop = lm_fit(X=design, y=props)
        # z = np.array(list(map(lambda x: x**contrasts ,fit_prop['coefficients']))).T
        z = (fit_prop['coefficients']**contrasts).T
        RR = np.prod(z, axis=0)
    # If confounding variables included in design matrix exclude them
    else:
        design = design.iloc[:, np.where(contrasts != 0)[0]]
        fit_prop = lm_fit(X=design, y=props)
        new_cont = contrasts[contrasts != 0]
        z = (fit_prop['coefficients']**new_cont).T
        RR = np.prod(z, axis=0)

    # adjust p_values using benjamin hochberg method
    p_values = fit_cont['p_value'].flatten()
    fdr = multipletests(p_values, method='fdr_bh')

    # save results to dictionary
    res = {}
    res['clusters'] = props.columns.to_list()
    for i, cond in enumerate(design.columns):
        res['mean_props_' + cond] = fit_prop['coefficients'].T[i]

    res['prop_ratio'] = RR
    res['t_statistics'] = fit_cont['t'].flatten()
    res['p_values'] = p_values
    res['adjusted_p_values'] = fdr[1]
    cols = list(res.keys())

    return pd.DataFrame(res, columns=cols).set_index('clusters')


def sim_scanpro(data, clusters_col, conds_col, samples_col=None,
                transform='arcsin', n_reps=8, n_sims=100,
                conditions=None, robust=True, verbose=True):
    """Run scanpro multiple times on same dataset and pool estimates together.

    :param anndata.AnnData or pandas.DataFrame data: Single cell data with columns containing sample,
    condition and cluster/celltype information.
    :param str clusters_col: Name of column in date or data.obs where cluster/celltype information are stored.
    :param str conds_col: Column in data or data.obs where condition informtaion are stored.
    :param str samples_col: Column in data or data.obs where sample informtaion are stored, defaults to None.
    :param str transform: Method of transformation of proportions, defaults to 'logit'.
    :param int n_reps: Number of replicates to simulate if data does not have replicates, defaults to 8.
    :param int n_sims: Number of simulations to perform if data does not have replicates, defaults to 100.
    :param str conditions: List of condtitions of interest to compare, defaults to None.
    :param bool robust: Robust ebayes estimation to mitigate the effect of outliers, defaults to True.
    :param bool verbose: defaults to True.
    :return scanpro: A scanpro object containing estimated mean proportions for each cluster
    and median p-values from all simulations.
    """
    # check datas type
    if type(data).__name__ == "AnnData":
        data = data.obs

    # check samples column
    if samples_col is None:
        # this information is given in the function calling sim_scanpro
        # if verbose:
        #     print("samples_col was not provided! conds_col will be set as samples_col")
        # copy dataframe
        data = data.copy()
        # add conds_col as samples_col
        samples_col = 'pseudo_samples'
        data[samples_col] = data[conds_col]

    # get list of conditions and number of conditions
    if conditions is None:
        conditions = data[conds_col].unique()
    n_conds = len(conditions)

    # get original counts and proportions
    counts, props, prop_trans = get_transformed_props(data, sample_col=samples_col,
                                                      cluster_col=clusters_col, transform=transform)
    # get original design matrix
    design = create_design(data=data, samples=samples_col, conds=conds_col, reindex=props.index)

    # initiate lists to save results
    res = {}
    n_clusters = len(data[clusters_col].unique())
    coefficients = {condition: np.zeros((n_sims, n_clusters)) for condition in conditions}
    p_values = np.zeros((n_sims, n_clusters))
    counts_list = []
    props_list = []
    prop_trans_list = []
    design_list = []

    if verbose:
        print(f'Generating {n_reps} replicates and running {n_sims} simulations...')

    # start timer
    start = time.time()
    for i in range(n_sims):
        # generate replicates
        rep_data = generate_reps(data=data, n_reps=n_reps, sample_col=samples_col)

        # run propeller
        try:
            out_sim = run_scanpro(rep_data, clusters=clusters_col, samples=samples_col,
                                  conds=conds_col, transform=transform,
                                  conditions=conditions, robust=robust, verbose=False)
        # workaround brentq error "f(a) and f(b) must have different signs"
        # rerun simulation instead of crashing
        except ValueError:
            i -= 1
            continue

        # save counts, props and prop_trans
        counts_list.append(out_sim.counts)
        props_list.append(out_sim.props)
        prop_trans_list.append(out_sim.prop_trans)
        design_list.append(out_sim.design)

        # get adjusted p values for simulation
        try:  # check if all clusters are simulated, if a cluster is missing, rerun simulation
            p_values[i] = out_sim.results.iloc[:, -1].to_list()
        except ValueError:
            i -= 1
            continue

        # get coefficients estimates from linear model fit
        for k, cluster in enumerate(out_sim.results.index):
            for j, condition in enumerate(conditions):
                coefficients[condition][i, k] = out_sim.results.iloc[k, j]
    # end timer
    end = time.time()
    elapsed = end - start
    if verbose:
        print(f"Finished {n_sims} simulations in {round(elapsed, 2)} seconds")

    # save design matrix
    # make sure to include all simulations as some clusters may be missing in some simulations
    design_sim = pd.concat(design_list)
    design_sim = design_sim[~design_sim.index.duplicated(keep='first')].sort_index()  # remove all but first occurence of index

    # combine coefficients
    combined_coefs = combine(fit=coefficients, conds=conditions, n_clusters=n_clusters, n_conds=n_conds, n_sims=n_sims)

    # get mean counts, proportions and transformed proportions from all simulations
    counts_mean = get_mean_sim(counts_list)
    props_mean = get_mean_sim(props_list)
    prop_trans_mean = get_mean_sim(prop_trans_list)

    # get clusters names
    res['clusters'] = list(counts.columns)
    for i, condition in enumerate(coefficients.keys()):
        res['mean_props_' + condition] = combined_coefs[i]
    # calculate median of p values from all runs
    res['p_values'] = np.median(p_values, axis=0)

    # create dataframe for results
    out = pd.DataFrame(res).set_index('clusters')

    # create scanpro object
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Pandas doesn't allow columns to be created via a new attribute name*")

        output_obj = ScanproResult()
        output_obj.results = out
        output_obj.counts = counts  # original counts
        output_obj.sim_counts = counts_mean  # mean of all simulated counts
        output_obj.props = props
        output_obj.sim_props = props_mean
        output_obj.prop_trans = prop_trans
        output_obj.sim_prop_trans = prop_trans_mean
        output_obj.design = design
        output_obj.sim_design = design_sim
        output_obj.conditions = conditions

    # remove temporary samples column
    if samples_col is None:
        data.drop(samples_col, axis=1, inplace=True)

    return output_obj
