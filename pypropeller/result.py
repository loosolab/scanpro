""" A class containing the results of a pypropeller analysis """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


#class PyproResult(pd.DataFrame):
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
             type='all',
             clusters=None,
             n_columns=3):

        if clusters is None:
            clusters = self.props.columns.tolist()
        else:
            pass  # TODO: check if clusters are in the index

        sample_col = self.design.index.name

        prop_merged = self._merge_design_props()

        # Create a figure with n_columns
        n_columns = min(n_columns, len(clusters))  # number of columns are at least the number of clusters
        n_rows = math.ceil(len(clusters) / n_columns)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(3 * n_columns, 3 * n_rows))

        # Fill in the plot
        axes = axes.flatten() if len(clusters) > 1 else [axes]
        for i, cluster in enumerate(clusters):

            # Show the legend for the last plot on first row:
            if i == n_columns - 1:
                legend = True
            else:
                legend = False

            # Plot the proportions
            if type == 'stripplot':
                ax = sns.stripplot(data=prop_merged, y=cluster, x="Group", hue=sample_col, legend=legend, jitter=True, ax=axes[i])
            elif type == 'boxplot':
                ax = sns.boxplot(data=prop_merged, y=cluster, x="Group", color="white", showfliers=False, ax=axes[i])
            elif type == 'barplot':
                ax = sns.barplot(data=prop_merged, y=cluster, x="Group", hue=sample_col, ax=axes[i])
                ax.legend_.remove()

            ax.set_title(cluster)
            ax.set(ylabel='Proportions')

        fig.tight_layout()

        # Add legend to the last plot
        if not type == 'boxplot':
            axes[n_columns - 1].legend(title=sample_col, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

        # Remove empty plots
        for i in range(len(clusters), len(axes)):
            axes[i].set_visible(False)
