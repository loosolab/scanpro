""" A class containing the results of a scanpro analysis """

import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from statannotations.Annotator import Annotator


class ScanproResult():

    @property
    def _constructor(self):
        return ScanproResult

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
             simulated=False,
             save=False):
        """Plot proportions pro condition

        :param str kind: Kind of plot (stripplot, barplot and boxplot), defaults to 'stripplot'
        :param list or str clusters: Specify clusters to plot, if None, all clusters will be plotted, defaults to None
        :param int n_columns: Number of columns in the figure, defaults to 3
        :param bool simulated: If True, simulated results will be plotted, defaults to False
        :param str save: Path to save plot, add extension at the end e.g. 'path/to/file.png', defaults to False
        """
        # get results dataframe
        results = self.results if not simulated else self.sim_results

        # if no clusters are specified, plot all clusters
        if clusters is None:
            clusters = self.props.columns.tolist()
            # get all p_values
            p_values = round(results.iloc[:,-1], 3).to_list()
        else:
            if not isinstance(clusters, list):
                clusters = [clusters]
            # check if provided clusters are in data
            check = all(item in self.props.columns for item in clusters)
            if not check:
                s1 = "The following clusters could not be found in data: "
                s2 = ', '.join([clusters[i] for i in np.where(np.isin(clusters, self.props.columns, invert=True))[0]])
                raise ValueError(s1 + s2)
            # get p_values of specified clusters
            p_values = round(results.loc[clusters].iloc[:,-1], 3).to_list()

        sample_col = self.design.index.name if not simulated else self.sim_design.index.name
        n_conds = len(self.design.columns)
        prop_merged = self._merge_design_props(simulated=simulated)


        # Create a figure with n_columns
        n_columns = min(n_columns, len(clusters))  # number of columns are at least the number of clusters
        n_rows = math.ceil(len(clusters) / n_columns)
        width = n_conds // 2 if n_conds > 8 else 3
        hight = (n_conds // 2) + 1 if n_conds > 8 else 4

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(width * n_columns, hight * n_rows))

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
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            # get p-value as string
            p_value = f"p={p_values[i]}"
            # check number of compared conditions
            n_compared_conds = len(self.conditions)
            if n_compared_conds == 2:
                # pairs to plot p values
                #all_conds = True if len(self.conditions) == len(self.design.columns) else False  # check if all conditions were chosen
                pairs = [(self.conditions[0], self.conditions[-1])]
                line_width = 1.5
            else:
                # get x axis labels from plot
                labels = [label.get_text() for label in ax.get_xticklabels()]
                # choose first and last label to have p-value in the centre
                pairs = [(labels[0], labels[-1])]
                # if more than 2 conditions, don't plot horizontal bar
                line_width = 0

            # add p values to plot
            annot = Annotator(ax, pairs=pairs, data=prop_merged, y=cluster, x="Group", verbose=False)
            (annot
             .configure(test=None, line_width=line_width, verbose=False)
             #.set_pvalues(pvalues=[p_values[i]])
             .set_custom_annotations([p_value])
             .annotate())

        plt.subplots_adjust(wspace=0.5, hspace=0.6)

        # Add legend to the last plot
        if not kind == 'boxplot':
            axes[n_columns - 1].legend(title=sample_col, bbox_to_anchor=(1.05, 1), loc="upper left",
                                       frameon=False, ncols=2)

        # Remove empty plots
        for i in range(len(clusters), len(axes)):
            axes[i].set_visible(False)

        if save:
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')

    def plot_samples(self, stacked=True,
                     x='samples',
                     save=False):
        """Plot proportions of clusters pro sample

        :param bool stacked: If True, a stacked bar plot is plotted, defaults to True
        :param str x: Specifies if clusters or samples are plotted as x axis, defaults to 'samples'
        :param str save: Path to save plot, add extension at the end e.g. 'path/to/file.png', defaults to False
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

        if save:
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')
