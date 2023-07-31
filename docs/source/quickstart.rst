Scanpro quickstart
========================

To use Scanpro, import the module and run the scanpro function.

.. code-block:: python

    from scanpro.scanpro import scanpro

    out = scanpro(adata, samples_col='sample', clusters_col='cluster', conds_col='conds')

    out.results

To plot results, run

.. code-block:: python

    out.plot()

    # For boxplots, use the parameter <kind>

    out.plot(kinde='boxplot')
