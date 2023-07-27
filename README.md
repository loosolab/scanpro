# Scanpro: robust proportion analysis for single cell resolution data
Scanpro offers a python implementation of the propeller method [(Phipson et al., 2022)](https://academic.oup.com/bioinformatics/article/38/20/4720/6675456) to test the significance of changes in cell proportions
across different conditions from single cell clustering data. Scanpro also supports datasets without replicates by simulating artificial replicates using bootstrapping, and integrates seamlessly into existing frameworks using the AnnData format.

<img src="docs/source/figures/scanpro_workflow.png" width=75% height=55%>

## Install
To install scanpro: 
- clone the repository
```
git clone https://gitlab.gwdg.de/loosolab/software/scanpro.git
```
- navigate to scanpro directory
```
cd scanpro
```
- then run 
```
pip install .
```
## Quick start
To run the tool import and call the function `scanpro`:
```
from scanpro import scanpro

out = scanpro.scanpro(adata, clusters_col='clusters', conds_col='condition', samples_col='sample')

```

- If samples_col is not given or set to None, the dataset is assumed to be not replicated and scanpro will run the bootstrapping method.

To show the results, run
```out.results```. 

You can plot the results by calling ```out.plot()```.
