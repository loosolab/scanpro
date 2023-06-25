""" A class containing the results of a scanpro analysis """

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
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

        # Get the design and proportions
        design = self.sim_design if simulated else self.design
        props = self.sim_props if simulated else self.props

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
             save=False):
        """Plot proportions pro condition

        :param str kind: Kind of plot (stripplot, barplot and boxplot), defaults to 'stripplot'
        :param list or str clusters: Specify clusters to plot, if None, all clusters will be plotted, defaults to None
        :param int n_columns: Number of columns in the figure, defaults to 3
        :param bool simulated: If True, simulated results will be plotted, defaults to False
        :param str save: Path to save plot, add extension at the end e.g. 'path/to/file.png', defaults to False
        """
        # get results dataframe
        simulated = True if hasattr(self, "sim_results") else False
        results = self.sim_results if simulated else self.results
        design = self.sim_design if simulated else self.design

        # if no clusters are specified, plot all clusters
        all_clusters = self.props.columns.tolist()
        if clusters is None:
            clusters = all_clusters
            # get all p_values
            p_values = round(results.iloc[:, -1], 3).to_list()
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
            p_values = round(results.loc[clusters].iloc[:, -1], 3).to_list()   # the last column contains the p_values

        sample_col = design.index.name
        n_conds = len(self.design.columns)
        prop_merged = self._merge_design_props()
        prop_merged["simulated"] = False

        # If the results contain simulated proportions, add these proportions to prop_merged
        if simulated:
            prop_merged_simulated = self._merge_design_props(simulated=True)
            prop_merged_simulated["simulated"] = True
            prop_merged = pd.concat([prop_merged, prop_merged_simulated])

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

            # Plot the proportions to axis
            ax = axes[i]
            ax2 = None  # for simulated data in stripplot
            if kind == 'stripplot':

                sns.stripplot(data=prop_merged, y=cluster, x="group", jitter=True, ax=ax, alpha=0)  # initialize by plotting invisible points (alpha=0) to get the full axes limits

                for simulated_bool, prop_table in prop_merged.sort_values("simulated").groupby("simulated"):  # sortby ensures that original data is plotted first
                    if simulated_bool:
                        marker = "s"  # square marker for simulated data
                        sample2marker = {sample: marker for sample in prop_table[sample_col]}  # for adjusting legend later

                        # Create second axes to enable second legend
                        ax2 = ax.twinx()
                        ax2.set_yticks([])  # remove yticks
                        sns.stripplot(data=prop_table, y=cluster, x="group", hue=sample_col, legend=legend, jitter=True, ax=ax2, marker=marker, size=7)

                    else:
                        sns.stripplot(data=prop_table, y=cluster, x="group", hue=sample_col, legend=legend, jitter=True, ax=ax, marker="o", size=7)

            elif kind == 'boxplot':
                prop_table = prop_merged[prop_merged["simulated"]] if simulated else prop_merged  # if simulated = True, only show simulated data
                sns.boxplot(data=prop_table, y=cluster, x="group", color="white", showfliers=False, ax=ax)
            elif kind == 'barplot':
                prop_table = prop_merged[prop_merged["simulated"]] if simulated else prop_merged  # if simulated = True, only show simulated data
                sns.barplot(data=prop_table, y=cluster, x="group", hue=sample_col, ax=ax)
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
                pairs = [(self.conditions[0], self.conditions[-1])]
                line_width = 1.5
            else:
                # get x axis labels from plot
                labels = [label.get_text() for label in ax.get_xticklabels()]
                # choose first and last label to have p-value in the centre
                pairs = [(labels[0], labels[-1])]
                # if more than 2 conditions, don't plot horizontal bar
                line_width = 0

            # get y-axis with maximum value
            ax_p = ax
            if ax2:
                ax_p = ax2 if ax2.get_ylim()[1] > ax.get_ylim()[1] else ax

            # add p values to plot
            annot = Annotator(ax_p, pairs=pairs, data=prop_merged, y=cluster, x="group", verbose=False)
            (annot
             .configure(test=None, line_width=line_width, verbose=False)
             .set_custom_annotations([p_value])
             .annotate())

            # If stripplot and simulated, adjust y limit to be the same for both axes (original and simulated data)
            if ax2 is not None:
                ax2.set_ylim(ax.get_ylim())

            # Add legend to the last plot
            if legend and kind != "boxplot":

                # Plot first legend
                l1 = ax.legend(title=sample_col, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, ncols=2)

                if kind == "stripplot" and simulated:

                    # Plot second legend to get size
                    l2 = ax2.legend(ncols=2)

                    # get extent of both legends
                    l1_extent = l1.get_window_extent().transformed(ax.transAxes.inverted())
                    l2_extent = l2.get_window_extent().transformed(ax.transAxes.inverted())

                    # adjust location of second legend
                    l2_height = (l2_extent.y1 - l2_extent.y0) * 1.2  # get second legend height + extent a little to make room
                    l2_loc = (l1_extent.x0, l1_extent.y0 - l2_height)  # location is lower left corner

                    # Adjust marker handles manually (bug in legend shows all legends as circles)
                    handles, labels = ax2.get_legend_handles_labels()
                    handles = [Line2D([], [], color=h.get_facecolor(), linestyle='', marker=sample2marker[l])
                               for h, l in zip(handles, labels)]

                    # Final legend location
                    ax2.legend(handles, labels, title="Simulated replicates", frameon=False, ncols=2, loc=l2_loc)

        plt.subplots_adjust(wspace=0.5, hspace=0.6)

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

        if x == 'samples':
            ax = self.props.plot(kind='bar', stacked=stacked)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="clusters")
        elif x == 'clusters':
            ax = self.props.T.plot(kind='bar', stacked=stacked)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="samples")

        # set labels
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel("Proportions")

        if save:
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')
