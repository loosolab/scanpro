""" A class containing the results of a scanpro analysis """

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np


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
        design_melt = design.reset_index().melt(id_vars=sample_col, var_name=self.conds_col)
        design_melt = design_melt[design_melt["value"] == 1]
        design_melt = design_melt.drop(columns="value")
        design_melt = design_melt.loc[:, ~design_melt.columns.duplicated()].copy()  # remove duplicate columns, e.g. if sample_col == conds_col

        # Subset conditions to those in all_conditions (to not include covariates in the plot)
        design_melt = design_melt[design_melt[self.conds_col].isin(self.all_conditions)]

        if self.covariates:
            design_melt.drop_duplicates(inplace=True)

        # Merge the proportions with the design matrix
        prop_merged = props.reset_index().merge(design_melt, left_on=sample_col, right_on=sample_col, how="left")
        prop_merged.index = props.index

        return prop_merged

    def plot(self,
             kind='stripplot',
             clusters=None,
             n_columns=3,
             figsize=None,
             show=True,
             save=False):
        """Plot proportions pro condition

        :param str kind: Kind of plot (stripplot, barplot and boxplot), defaults to 'stripplot'
        :param list or str clusters: Specify clusters to plot, if None, all clusters will be plotted, defaults to None
        :param int n_columns: Number of columns in the figure, defaults to 3
        :param tuple figsize: Figure size, defaults to None (size is automatic)
        :param str save: Path to save plot, add extension at the end e.g. 'path/to/file.png', defaults to False
        """

        # get results dataframe
        simulated = True if hasattr(self, "sim_results") else False
        results = self.sim_results if simulated else self.results
        design = self.sim_design if simulated else self.design
        conds_col = self.conds_col

        # drop covariates from design matrix
        if self.covariates:
            design = design[self.all_conditions]

        # if no clusters are specified, plot all clusters
        if clusters is None:
            clusters = self.results.index.tolist()
        else:
            if not isinstance(clusters, list):
                clusters = [clusters]

            # check if provided clusters are in data
            check = all(item in self.props.columns for item in clusters)
            if not check:
                s1 = "The following clusters could not be found in data: "
                s2 = ', '.join([clusters[i] for i in np.where(np.isin(clusters, self.props.columns, invert=True))[0]])
                raise ValueError(s1 + s2)

        # get p_values of clusters
        p_values = results.loc[clusters].iloc[:, -1].to_list()   # the last column contains the p_values
        p_values_fmt = [str(round(p, 3)) if p > 0.001 else f"{p:.2e}" for p in p_values]  # format to scientific notation for small p-values
        p_values_fmt = ["0.0" if p == "0.00e+00" else p for p in p_values_fmt]  # format 0-values

        # Collect props for original data
        sample_col = design.index.name
        n_conds = len(self.design.columns)
        prop_merged = self._merge_design_props()
        prop_merged["simulated"] = False

        # If the results contain simulated proportions, add these proportions to prop_merged
        if simulated:
            prop_merged[sample_col] = prop_merged.index

            prop_merged_simulated = self._merge_design_props(simulated=True)
            prop_merged_simulated[sample_col] = prop_merged_simulated.index
            prop_merged_simulated["simulated"] = True

            prop_merged = pd.concat([prop_merged, prop_merged_simulated])

        # Sort the dataframe by the order of the conditions in the design matrix
        prop_merged.reset_index(drop=True, inplace=True)
        column_order = {col: i for i, col in enumerate(self.design.columns)}
        prop_merged["_order"] = prop_merged[conds_col].map(column_order)
        prop_merged.sort_values(["_order", sample_col], inplace=True)  # first by condition order, then sample order

        # Create a figure with n_columns
        n_columns = min(n_columns, len(clusters))  # number of columns are at least the number of clusters
        n_rows = math.ceil(len(clusters) / n_columns)
        width = n_conds // 2 if n_conds > 8 else 3.5
        hight = (n_conds // 2) + 1 if n_conds > 8 else 4.5

        if figsize is None:
            figsize = (width * n_columns, hight * n_rows)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=figsize)

        # Fill in the plot
        axes = axes.flatten() if len(clusters) > 1 else [axes]
        axes2 = []
        for i, cluster in enumerate(clusters):

            # Show the legend for the last plot on first row:
            if i == n_columns - 1:
                legend = True
            else:
                legend = False

            # Plot the proportions to axis
            x_order = prop_merged[conds_col].unique()
            ax = axes[i]
            ax2 = None  # for simulated data in stripplot
            if kind == 'stripplot':

                sns.stripplot(data=prop_merged, y=cluster, x=conds_col, jitter=True, ax=ax, alpha=0)  # initialize by plotting invisible points (alpha=0) to get the full axes limits

                for simulated_bool in prop_merged["simulated"].unique():
                    prop_table = prop_merged[prop_merged["simulated"] == simulated_bool]

                    if simulated_bool:
                        prop_table = prop_table.sort_index()  # to be compatible with original data
                        marker = "s"  # square marker for simulated data

                        # Create second axes to enable second legend
                        ax2 = ax.twinx()
                        ax.set_zorder(ax2.get_zorder() + 1)  # put original ax in front of ax2
                        ax.set_frame_on(False)               # ensure that ax2 is still visible behind ax

                        ax2.set_ylim(ax.get_ylim())  # set ylims to be the same as in the first axes
                        ax2.set_yticks([])  # remove yticks
                        ax2.set_ylabel("")  # remove ylabel
                        axes2.append(ax2)
                        sns.stripplot(data=prop_table, y=cluster, x=conds_col, hue=sample_col, jitter=True, ax=ax2,
                                      marker=marker, size=7, order=x_order, legend=legend)

                    else:
                        lw = 1 if simulated else 0   # original data is has a border, simulated data does not
                        color = "black" if simulated else None  # original data is black, simulated data is colored
                        sns.stripplot(data=prop_table, y=cluster, x=conds_col, hue=sample_col, jitter=True, ax=ax,
                                      marker="o", size=7, linewidth=lw, color=color, order=x_order, legend=legend)

            elif kind == 'boxplot':
                prop_table = prop_merged[prop_merged["simulated"]] if simulated else prop_merged  # if simulated = True, only show simulated data
                sns.boxplot(data=prop_table, y=cluster, x=conds_col, color="white", showfliers=False, ax=ax)

            elif kind == 'barplot':
                prop_table = prop_merged[prop_merged["simulated"]] if simulated else prop_merged  # if simulated = True, only show simulated data
                sns.barplot(data=prop_table, y=cluster, x=conds_col, hue=sample_col, ax=ax)
                ax.legend_.remove()

            ax.set_title(cluster)
            ax.set(ylabel='Proportions')
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            # check number of compared conditions
            n_conds = len(self.all_conditions)
            n_compared_conds = len(self.conditions)
            labels = [label.get_text() for label in ax.get_xticklabels() if label.get_text() in self.conditions]
            labels_idx = [j for j, label in enumerate(ax.get_xticklabels()) if label.get_text() in self.conditions]

            # get p-value as string
            p_value = f"p={p_values_fmt[i]}"

            # plot bracket for compared conditions
            if n_compared_conds == n_conds:  # if comparing all conditions,  don't plot horizontal bar
                line_width = 0

            else:
                line_width = 1.5

            # get y-axis with maximum value plotted
            max_prop = prop_merged.groupby("simulated")[cluster].max().sort_values(ascending=False)
            sim_bool_max = max_prop.index[0]
            ax_p = ax
            if ax2 is not None:
                if sim_bool_max:  # if simulated has the highest value, use ax2 for p-values
                    ax_p = ax2

            # add p values to plot
            x1 = labels_idx.pop(0)
            x2 = labels_idx.pop(-1)
            y, h, col = ax_p.get_ylim()[1] + (ax_p.get_ylim()[1] * 0.03), (ax_p.get_ylim()[1] - ax_p.get_ylim()[0]) * 0.05, 'k'

            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=line_width, c=col)
            ax.text((x1 + x2) * 0.5, y + (h + h * 0.1), p_value, ha='center', va='bottom', color=col, fontsize='medium')
            for x in labels_idx:
                ax.plot([x1, x1, x, x], [y, y + h, y + h, y], lw=line_width, c=col)

            # If stripplot and simulated, adjust y limit to be the same for both axes (original and simulated data)
            if ax2 is not None:
                ax_ylim = ax.get_ylim()
                ax2_ylim = ax2.get_ylim()
                new_ylim = (min(ax_ylim[0], ax2_ylim[0]), max(ax_ylim[1], ax2_ylim[1]))

                ax.set_ylim(new_ylim)
                ax2.set_ylim(new_ylim)
                ax2.set_ylabel("")

            # add extra space on plot for the annotation
            ax.set_ylim(top=ax.get_ylim()[1] + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04))
            ax.set_xlim(left=ax.get_xlim()[0] - ((ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05),
                        right=ax.get_xlim()[1] + ((ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05))

        plt.subplots_adjust(wspace=0.5, hspace=0.6)

        # Add legend to the last plot of first row
        if kind != "boxplot":

            # Plot first legend
            ax = axes[n_columns - 1]
            legend_title = sample_col if not simulated else conds_col    # for original data is samples == conditions if simulated
            l1 = ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, ncols=2)

            # Plot legend for simulated data
            if kind == "stripplot" and simulated:

                # Plot second legend to get size
                ax2 = axes2[n_columns - 1]
                l2 = ax2.legend(ncols=2)

                # get extent of both legends
                l1_extent = l1.get_window_extent().transformed(ax.transAxes.inverted())
                l2_extent = l2.get_window_extent().transformed(ax.transAxes.inverted())

                # adjust location of second legend
                l2_height = (l2_extent.y1 - l2_extent.y0) * 1.2  # get second legend height + extent a little to make room
                l2_loc = (l1_extent.x0, l1_extent.y0 - l2_height)  # location is lower left corner

                # Adjust marker handles manually (bug in legend shows all legends as circles)
                handles, labels = ax2.get_legend_handles_labels()
                handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))  # sort by label name

                # Final legend location
                ax2.legend(handles, labels, title="Simulated replicates", frameon=False, ncols=2, loc=l2_loc)

        # Remove empty plots
        for i in range(len(clusters), len(axes)):
            axes[i].set_visible(False)

        if save:
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')

        if not show:
            return ax

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
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title=self.props.index.name, frameon=False)
        elif x == 'clusters':
            ax = self.props.T.plot(kind='bar', stacked=stacked)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title=self.props.T.index.name, frameon=False)

        # set labels
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel("Proportions")

        if save:
            plt.savefig(fname=save, dpi=600, bbox_inches='tight')
