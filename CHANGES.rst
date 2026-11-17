0.4.1 (17.11.2026)
-------------------
- Fix filling of missing values for clusters with 0-counts, which had no effect with pandas >= 3.0 (Copy-on-Write)
- Fix KeyError in convert_counts_to_df with pandas >= 3.0
- Fix KeyError in simulate_cell_counts_2 with pandas >= 3.0 when beta parameters are given as pandas.Series
- Remove cells with missing cluster, condition, sample or covariate information with a warning (in scanpro and run_scanpro). Previously, missing clusters resulted in an empty NaN row in the results and wrong baseline proportions
- Fix arguments not passed on in scanpro: n_sims was ignored with pairwise=False, verbosity was ignored with pairwise=True
- Fix failed simulations in sim_scanpro being skipped instead of rerun, which resulted in fewer than n_sims simulations. An error is raised if more than n_sims simulations fail
- Remove unused calculation of coefficients in sim_scanpro

0.4.0 (03.09.2025)
-------------------
- Add pairwise testing: test each pair of conditions individually
- Remove numpy version restriction
- Refactor main module

0.3.2 (01.08.2024)
-------------------
- Replace Fortran implementation of gausq2 function with python
- Replace deprecated numpy.distutils

0.3.1 (10.07.2024)
-------------------
- Remove statannotations package. Plotting p-values is now implemented directly

0.3 (28.06.2024)
-------------------
- Add covariate testing with unreplicated datasets
- Add automatic selection for number of pseudo-replicates


0.2 (14.08.2023)
-------------------
- Fix wrong order of conditions in pseudoreplicate stripplot
- Added more information to setup.py, e.g. readme and url.
- Added figsize to ScanproResult.plot() to be able to change the aspect ratios of the plots.


0.1 (31.07.2023)
--------------------
- Initial release to PyPi
