""" A class containing the results of a pypropeller analysis """

import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from statannotations.Annotator import Annotator


class PyproResult():

    @property
    def _constructor(self):
        return PyproResult

    def _merge_design_props(self, simulated=False):
        """Merge proportions matrix with design matrix for plotting

        :param bool simulated: True if results were simulated
        :return pandas.DataFrame: Merged proportions and design
        """

        design = self.design if not simulated else self.sim_design
        props = self.props if not simulated else self.sim_props

        # Establish the samples per group
        sample_col = design.index.name
        design_melt = design.reset_index().melt(id_vars=sample_col)
        design_melt = design_melt[design_melt["value"] == 1]
        design_melt = design_melt.drop(columns="value")

        # Merge the proportions with the design matrix
        prop_merged = props.merge(design_melt, left_index=True, right_on=sample_col)

        return prop_merged

    def plot(self,
             kind='stripplot',
             clusters=None,
             n_columns=3,
             simulated=False):
        """Plot proportions pro condition

        :param str kind: Kind of plot (stripplot, barplot and boxplot), defaults to 'stripplot'
        :param list or str clusters: _description_, defaults to None
        :param int n_columns: _description_, defaults to 3
        :param bool simulated: True if results were simulated
        """
        # if no clusters are specified, plot all clusters
        if clusters is None:
            clusters = self.props.columns.tolist()
        else:
            if not isinstance(clusters, list):
                clusters = [clusters]
            # check if provided clusters are in data
            check = all(item in self.props.columns for item in clusters)
            if not check:
                s1 = "The following clusters could not be found in data: "
                s2 = ', '.join([clusters[i] for i in np.where(np.isin(clusters, self.props.columns, invert=True))[0]])
                raise ValueError(s1 + s2)

        sample_col = self.design.index.name if not simulated else self.sim_design.index.name
        prop_merged = self._merge_design_props(simulated=simulated)

        # get results dataframe
        results = self.results if not simulated else self.sim_results

        # Create a figure with n_columns
        n_columns = min(n_columns, len(clusters))  # number of columns are at least the number of clusters
        n_rows = math.ceil(len(clusters) / n_columns)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(3 * n_columns, 4 * n_rows))

        # Fill in the plot
        axes = axes.flatten() if len(clusters) > 1 else [axes]
        for i, cluster in enumerate(clusters):

            # Show the legend for the last plot on first row:
            if i == n_columns - 1:
                legend = True
            else:
                legend = False

            # Plot the proportions
            if kind == 'stripplot':
                ax = sns.stripplot(data=prop_merged, y=cluster, x="Group", hue=sample_col, legend=legend, jitter=True, ax=axes[i])
            elif kind == 'boxplot':
                ax = sns.boxplot(data=prop_merged, y=cluster, x="Group", color="white", showfliers=False, ax=axes[i])
            elif kind == 'barplot':
                ax = sns.barplot(data=prop_merged, y=cluster, x="Group", hue=sample_col, ax=axes[i])
                ax.legend_.remove()

            ax.set_title(cluster)
            ax.set(ylabel='Proportions')
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')

            if self.conditions is not None:
                pairs = (self.conditions[0], self.conditions[-1])
            else:
                pairs = (prop_merged.Group.unique()[0], prop_merged.Group.unique()[-1])
            pairs = [pairs]
            annot = Annotator(ax, pairs=pairs, data=prop_merged, y=cluster, x="Group", verbose=False)
            (annot
             .configure(test=None, verbose=False)
             .set_pvalues(pvalues=[round(results.iloc[i, -1], 3)])
             .annotate())

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Add legend to the last plot
        if not kind == 'boxplot':
            axes[n_columns - 1].legend(title=sample_col, bbox_to_anchor=(1.05, 1), loc="upper left",
                                       frameon=False, ncols=2)

        # Remove empty plots
        for i in range(len(clusters), len(axes)):
            axes[i].set_visible(False)

    def plot_samples(self, stacked=True,
                     x='samples'):
        """Plot proportions of clusters pro sample

        :param bool stacked: If True, a stacked bar plot is plotted, defaults to True
        :param str x: Specifies if clusters or samples are plotted as x axis, defaults to 'samples'
        """
        if stacked:
            if x == 'samples':
                self.props.plot(kind='bar', stacked=True).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            elif x == 'clusters':
                self.props.T.plot(kind='bar', stacked=True).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        else:
            if x == 'samples':
                self.props.plot.bar().legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            elif x == 'clusters':
                self.props.T.plot.bar().legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
