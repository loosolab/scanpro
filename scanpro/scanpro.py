import time
import warnings
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant

from scanpro.get_transformed_props import get_transformed_props
from scanpro.linear_model import lm_fit, contrasts_fit, create_design
from scanpro import ebayes
from scanpro.sim_reps import generate_reps, get_mean_sim
from scanpro.result import ScanproResult
from scanpro.logging import ScanproLogger


def scanpro(data, clusters_col, conds_col,
            samples_col=None,
            covariates=None,
            conditions=None,
            transform='logit',
            robust=True,
            n_sims=100,
            n_reps='auto',
            run_partial_sim=True,
            verbosity=1,
            seed=1):
    """Wrapper function for scanpro. The data must have replicates,
    since propeller requires replicated data to run. If the data doesn't have
    replicates, the function {sim_scanpro} will generate artificial replicates
    using bootstrapping and run propeller multiple times. The values are then pooled
    to get robust estimation of p values.

    :param anndata.AnnData or pandas.DataFrame data: Single cell data with columns containing sample,
        condition and cluster/celltype information.
    :param str clusters_col: Name of column in date or data.obs where cluster/celltype information are stored.
    :param str conds_col: Column in data or data.obs where condition information are stored.
    :param str samples_col: Column in data or data.obs where sample information are stored,
        if None, dataset is assumed to be not replicated and conds_col will be set as samples_col, defaults to None.
    :param list covariates: List of covariates to include in the model, defaults to None.
    :param str transform: Method of transformation of proportions, defaults to 'logit'.
    :param str conditions: List of condtitions of interest to compare, defaults to None.
    :param bool robust: Robust ebayes estimation to mitigate the effect of outliers, defaults to True.
    :param int n_sims: Number of simulations to perform if data does not have replicates, defaults to 100.
    :param int n_reps: Number of replicates to simulate if data does not have replicates,
        'auto' will generate pseudo-replicates for each sample based on its cell count,
        (3 for #cells<5000, 5 for #cells<14000 and 8 for #cells>14000), defaults to 'auto'.
    :param bool run_partial_sim: If True, the bootstrapping method will be also performed on datasets that are
        partially replicated (where some samples have replicates).
    :param int verbosity: Verbosity level for logging progress. 0=silent, 1=info, 2=debug. Defaults to 1.
    :param int seed: Seed for random number generator, defaults to 1.

    :raises ValueError: Data must have at least two conditions!
    :return ScanproResult: A scanpro object containing estimated mean proportions for each cluster and p-values.
    """

    logger = ScanproLogger(verbosity)  # create logger instance
    np.random.seed(seed)  # set seed for reproducibility (only relevant for simulated data)

    # Data must be Anndata or dataframe
    if type(data).__name__ == "AnnData":
        data = data.obs
    else:
        if type(data).__name__ != "DataFrame":
            raise ValueError("Data must be an AnnData or DataFrame object.")
    data = data.copy()  # make sure original data is not modified

    # Check format of covariates
    if covariates is not None:
        if isinstance(covariates, str):
            covariates = [covariates]
        elif not isinstance(covariates, list):  # if not a string, must be list
            raise ValueError("covariates must be a list of strings.")

    # check if samples_col and conds_col are in data
    columns = [clusters_col, conds_col]
    columns += [samples_col] if samples_col is not None else []    # add samples_col if given
    columns += covariates if covariates is not None else []        # add covariates if given
    columns_not_in_data = [col for col in columns if col not in data.columns]
    if len(columns_not_in_data) > 0:
        s = "The following columns could not be found in data: "
        s += ', '.join(columns_not_in_data)
        raise ValueError(s)

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
        conditions = data[conds_col].unique().tolist()

    # check if there are 2 conditions or more
    if len(conditions) < 2:
        raise ValueError("There has to be at least two conditions to compare! Only one condition was found: " + str(conditions))

    # if samples_col is None, data is not replicated
    if samples_col is None:
        repd = False
        partially_repd = False

        # add conds_col as samples_col
        samples_col = conds_col

    # otherwise, assume data is replicated
    else:
        repd = True
        partially_repd = False

        # Check that sample names are unique across conditions; else change sample names to condition_sample
        sample_info = data[[conds_col, samples_col]].drop_duplicates()
        samples = sample_info[samples_col].unique()
        if len(samples) != len(sample_info):
            logger.warning("Sample names are not unique across conditions! Changing sample names to <condition>_<sample>.")
            data[samples_col] = data[conds_col].astype(str) + '_' + data[samples_col].astype(str)

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

    # ---------------- Run Scanpro depending on replicates -------------- #
    # check if there are no replicates
    if not repd:

        logger.info("Your data doesn't have replicates! Artificial replicates will be simulated to run scanpro.")
        if transform != "arcsin":
            logger.warning("Consider setting transform='arcsin', as this produces more accurate results for simulated data.")
        logger.info("Simulation may take some minutes...")

        # set number of pseudo replicates based on sample cell count
        if n_reps == 'auto':
            # get smallest cell count in all samples
            n = data.value_counts(samples_col).min()
            if n < 5000:
                n_reps = 3
            elif n < 14000:
                n_reps = 5
            else:
                n_reps = 8

        # set transform to arcsin, since it produces more accurate results for simulations
        out = sim_scanpro(data, n_reps=n_reps, n_sims=n_sims, clusters_col=clusters_col, covariates=covariates,
                          conds_col=conds_col, transform=transform,
                          conditions=conditions, robust=robust, verbosity=verbosity)

    # if at least on condition doesn't have replicate, merge samples and bootstrap
    elif partially_repd:
        s = "The following conditions don't have replicates:  "
        s += ", ".join(no_reps_list)
        logger.info(s)
        if not run_partial_sim:
            logger.info("Normal scanpro will be performed. To also run Bootstrapping, set run_partial_sim=True")

        # add conditions as merged_samples column
        merged_samples_col = 'merged_samples'
        data[merged_samples_col] = data[conds_col]

        # set number of pseudo replicates based on sample cell count
        if n_reps == 'auto':
            # get smallest cell count in all samples
            n = data.value_counts(merged_samples_col).min()
            if n < 5000:
                n_reps = 3
            elif n < 14000:
                n_reps = 5
            else:
                n_reps = 8

        # run scanpro normally
        logger.info("Running scanpro with original replicates...")
        out = run_scanpro(data, clusters=clusters_col, samples=samples_col, conds=conds_col, covariates=covariates,
                          transform=transform, conditions=conditions, robust=robust, verbosity=verbosity)

        if run_partial_sim:
            # run simulations
            logger.info("Running scanpro with simulated replicates...")
            # set transform to arcsin, since it produces more accurate results for simulations
            transform = 'arcsin'
            out_sim = sim_scanpro(data, n_reps=n_reps, n_sims=n_sims, clusters_col=clusters_col, covariates=covariates,
                                  conds_col=conds_col, transform=transform,
                                  conditions=conditions, robust=robust, verbosity=verbosity)

            logger.info("To access results for original replicates, run <out.results>, and <out.sim_results> for simulated results")

    # if all conditions have replicates, run scanpro normally
    else:
        out = run_scanpro(data, clusters=clusters_col, samples=samples_col, conds=conds_col, covariates=covariates,
                          transform=transform, conditions=conditions, robust=robust, verbosity=verbosity)

    # Add additional clusters not included due to 0-counts in samples
    all_clusters = data[clusters_col].unique()
    missing_clusters = set(all_clusters) - set(out.results.index.unique())

    zero_rows = pd.DataFrame(np.nan, index=list(missing_clusters), columns=out.results.columns)
    out.results = pd.concat([out.results, zero_rows])
    for col in out.results.columns:
        if "p_values" in col:
            out.results[col].fillna(1, inplace=True)
        elif "mean_props" in col:
            out.results[col].fillna(0, inplace=True)
        elif "prop_ratio" in col:
            out.results[col].fillna(0, inplace=True)

    # add baseline proportions as first column
    baseline_props = data[clusters_col].value_counts() / data.shape[0]  # proportions of each cluster in all samples
    baseline_props = baseline_props.reindex(out.results.index)  # reindex to match order of clusters in out.results
    out.results.insert(0, 'baseline_props', baseline_props.values)  # put baseline_props first
    out.results.index.names = ['clusters']  # rename index column

    # Sort by p-values (small to large)
    out.results.sort_values(out.results.columns[-1])

    # add conditions to object
    out.conds_col = conds_col
    out.conditions = conditions
    out.covariates = covariates

    # if data is not replicated, add results also as sim_results for plotting
    if not repd:
        out.sim_results = out.results

    # add simulated results for partially replicated data
    if partially_repd and run_partial_sim:
        out.sim_results = out_sim.results
        out.sim_design = out_sim.sim_design
        out.sim_counts = out_sim.sim_counts
        out.sim_props = out_sim.sim_props
        out.sim_prop_trans = out_sim.sim_prop_trans

    return out


def run_scanpro(data, clusters, samples, conds, transform='logit',
                covariates=None, conditions=None, robust=True, verbosity=1):
    """Test the significance of changes in cell proportions across conditions in single-cell data. The function
    uses empirical bayes to moderate statistical tests to give robust estimation of significance.

    :param anndata.AnnData or pandas.DataFrame adata: Anndata object containing single-cell data.
    :param str clusters: Column in adata.obs where cluster or celltype information are stored.
    :param str samples: Column in adata.obs where sample information are stored.
    :param str conds: Column in adata.obs where condition information are stored.
    :param str transform: Method of normalization of proportions (logit or arcsin), defaults to 'logit'
    :param str conditions: List of condtitions of interest to compare, defaults to None.
    :param bool robust: Robust ebayes estimation to mitigate the effect of outliers, defaults to True
    :return ScanproResult: A scanpro object containing estimated mean proportions for each cluster and p-values.
    """

    logger = ScanproLogger(verbosity)

    # check data type
    if type(data).__name__ == "AnnData":
        data = data.obs

    # Infer missing arguments
    if covariates is None:
        covariates = []

    all_conditions = data[conds].unique().tolist()
    if conditions is None:
        conditions = all_conditions

    # calculate proportions and transformed proportions
    counts, props, prop_trans = get_transformed_props(data, sample_col=samples, cluster_col=clusters, transform=transform)

    # create design matrix
    design = create_design(data=data, sample_col=samples, conds_col=conds, covariates=covariates)

    # Subset design to specific conditions + covariates
    design_columns = [col for col in design.columns if col not in all_conditions or col in conditions]
    design_sub = design[design_columns]

    # Subset design and props to only included samples in conditions
    included_samples = design_sub[design_sub.sum(axis=1) != 0].index.tolist()
    design_sub = design_sub.loc[included_samples, :]
    props_sub = props.loc[included_samples, :]
    prop_trans_sub = prop_trans.loc[included_samples, :]

    # Remove celltypes not present in included samples
    nonzero_idx = props_sub.sum(axis=0) != 0
    props_sub = props_sub.loc[:, nonzero_idx]
    prop_trans_sub = prop_trans_sub.loc[:, nonzero_idx]

    # run t-test / anova
    if len(conditions) == 2:

        logger.info("There are 2 conditions. T-Test will be performed...")
        contrasts = np.zeros(len(design_sub.columns))
        contrasts[0] = 1
        contrasts[1] = -1
        # the rest of the columns in design will be used as covariates

        out = t_test(props_sub, prop_trans_sub, design_sub, contrasts, robust, verbosity)

    else:
        logger.info("There are more than 2 conditions. ANOVA will be performed...")
        coef = np.arange(len(conditions))
        out = anova(props_sub, prop_trans_sub, design_sub, coef, robust, verbosity)

    out.index.name = clusters
    logger.info("Done!")

    # create scanpro object
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Pandas doesn't allow columns to be created via a new attribute name*")

        output_obj = ScanproResult()
        output_obj.results = out
        output_obj.counts = counts
        output_obj.props = props
        output_obj.prop_trans = prop_trans
        output_obj.design = design
        output_obj.all_conditions = data[conds].unique().tolist()
        output_obj.conditions = conditions

    return output_obj


def anova(props, prop_trans, design, coef, robust=True, verbosity=1):
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

    logger = ScanproLogger(verbosity)

    # check if coef is a numpy array
    if not isinstance(coef, np.ndarray):
        coef = np.array(coef)

    # check if there are less than 3 clusters
    if prop_trans.shape[1] < 3 and robust:
        logger.info("Robust is set to True, but robust eBayes needs 3 or more clusters! Normal eBayes will be performed.")
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


def t_test(props, prop_trans, design, contrasts, robust=True, verbosity=1):
    """ Test the significance of changes in cell proportion across 2 conditions using
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

    logger = ScanproLogger(verbosity)

    # check if there are less than 3 clusters
    if prop_trans.shape[1] < 3 and robust:
        logger.info("Robust is set to True, but robust eBayes needs 3 or more clusters! Normal eBayes will be performed.")
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            z = (fit_prop['coefficients']**contrasts).T
        RR = np.prod(z, axis=0)

    # If confounding variables included in design matrix exclude them
    else:
        design = design.iloc[:, np.where(contrasts != 0)[0]]
        fit_prop = lm_fit(X=design, y=props)
        new_cont = contrasts[contrasts != 0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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


def sim_scanpro(data, clusters_col, conds_col,
                covariates=None,
                transform='arcsin', n_reps=8, n_sims=100,
                conditions=None, robust=True, verbosity=1):
    """Run scanpro multiple times on same dataset and pool estimates together.

    :param anndata.AnnData or pandas.DataFrame data: Single cell data with columns containing sample,
        condition and cluster/celltype information.
    :param str clusters_col: Name of column in date or data.obs where cluster/celltype information are stored.
    :param str conds_col: Column in data or data.obs where condition informtaion are stored.
    :param str transform: Method of transformation of proportions, defaults to 'logit'.
    :param int n_reps: Number of replicates to simulate if data does not have replicates, defaults to 8.
    :param int n_sims: Number of simulations to perform if data does not have replicates, defaults to 100.
    :param str conditions: List of condtitions of interest to compare, defaults to None.
    :param bool robust: Robust ebayes estimation to mitigate the effect of outliers, defaults to True.
    :param bool verbosity: Verbosity level, defaults to 1.
    :return ScanproResults: A ScanproResult object containing estimated mean proportions for each cluster
        and median p-values from all simulations.
    """

    logger = ScanproLogger(verbosity)

    # check datas type
    if type(data).__name__ == "AnnData":
        data = data.obs

    # get original counts and proportions
    counts, props, prop_trans = get_transformed_props(data, sample_col=conds_col,
                                                      cluster_col=clusters_col, transform=transform)

    # get original design matrix
    design = create_design(data=data, sample_col=conds_col, conds_col=conds_col, covariates=covariates)
    logger.info(f'Generating {n_reps} replicates and running {n_sims} simulations...')

    # start timer
    result_objects = []
    start = time.time()
    for i in range(n_sims):

        # generate replicates
        rep_data = generate_reps(data=data, n_reps=n_reps, sample_col=conds_col, covariates=covariates)
        samples_col = conds_col + "_replicates"

        # run propeller
        try:
            out_sim = run_scanpro(rep_data, clusters=clusters_col, samples=samples_col,
                                  conds=conds_col, covariates=covariates, transform=transform,
                                  conditions=conditions, robust=robust, verbosity=0)  # verbosity is 0 to prevent prints from individual simulations

        # workaround brentq error "f(a) and f(b) must have different signs"
        # rerun simulation instead of crashing
        except ValueError:
            i -= 1
            continue

        # save results object
        result_objects.append(out_sim)

    # end timer
    end = time.time()
    elapsed = end - start
    logger.info(f"Finished {n_sims} simulations in {round(elapsed, 2)} seconds")

    # Combine results from all simulations
    combined_results = pd.concat([result_object.results.reset_index() for result_object in result_objects])

    # Setup information for results
    prop_columns = [col for col in combined_results.columns if 'mean_props' in col]
    out = combined_results.groupby(clusters_col)[prop_columns + ["adjusted_p_values"]].median()
    out.rename(columns={"adjusted_p_values": "p_values"}, inplace=True)

    # Collect results from all simulations
    design_list = [obj.design for obj in result_objects]
    counts_list = [obj.counts for obj in result_objects]
    props_list = [obj.props for obj in result_objects]
    prop_trans_list = [obj.prop_trans for obj in result_objects]

    # get coefficients estimates from linear model fit
    n_clusters = len(data[clusters_col].unique())
    coefficients = {condition: np.zeros((n_sims, n_clusters)) for condition in conditions}
    for obj in result_objects:
        for k, cluster in enumerate(obj.results.index):
            for j, condition in enumerate(conditions):
                coefficients[condition][i, k] = obj.results.iloc[k, j]

    # save design matrix
    # make sure to include all simulations as some clusters may be missing in some simulations
    design_sim = pd.concat(design_list)
    design_sim = design_sim[~design_sim.index.duplicated(keep='first')].sort_index()  # remove all but first occurence of index

    # get mean counts, proportions and transformed proportions from all simulations
    counts_mean = get_mean_sim(counts_list)
    props_mean = get_mean_sim(props_list)
    prop_trans_mean = get_mean_sim(prop_trans_list)

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
        output_obj.all_conditions = data[conds_col].unique().tolist()
        output_obj.conditions = conditions

    # remove temporary samples column
    if samples_col is None:
        data.drop(samples_col, axis=1, inplace=True)

    return output_obj
