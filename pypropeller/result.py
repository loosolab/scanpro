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

    def _merge_design_props(self):
        """Merge proportions matrix with design matrix for plotting

        :return pandas.DataFrame: Merged proportions and design
        """

        # Establish the samples per group
        sample_col = self.design.index.name
        design_melt = self.design.reset_index().melt(id_vars=sample_col)
        design_melt = design_melt[design_melt["value"] == 1]
        design_melt = design_melt.drop(columns="value")

        # Merge the proportions with the design matrix
        prop_merged = self.props.merge(design_melt, left_index=True, right_on=sample_col)

        return prop_merged

    def plot(self,
             kind='stripplot',
             clusters=None,
             n_columns=3):
        """Plot proportions pro condition

        :param str kind: Kind of plot (stripplot, barplot and boxplot), defaults to 'stripplot'
        :param list or str clusters: _description_, defaults to None
        :param int n_columns: _description_, defaults to 3
        """
        if clusters is None:
            clusters = self.props.columns.tolist()
        else:
            if not isinstance(clusters, list):
                clusters = [clusters]
            # check if all clusters are in data
            check = all(item in self.props.columns for item in clusters)
            if not check:
                s1 = "The following clusters could not be found in data: "
                s2 = ', '.join([clusters[i] for i in np.where(np.isin(clusters, self.props.columns, invert=True))[0]])
                raise ValueError(s1 + s2)

        sample_col = self.design.index.name

        prop_merged = self._merge_design_props()

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
            pairs = [(prop_merged.Group.unique()[0], prop_merged.Group.unique()[-1])]
            annot = Annotator(ax, pairs=pairs, data=prop_merged, y=cluster, x="Group", verbose=False)
            (annot
             .configure(test=None, verbose=False)
             .set_pvalues(pvalues=[round(self.results.iloc[i, -1], 2)])
             .annotate())

        fig.tight_layout()

        # Add legend to the last plot
        if not kind == 'boxplot':
            axes[n_columns - 1].legend(title=sample_col, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

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
