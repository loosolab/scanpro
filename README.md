# Scanpro: robust proportion analysis for single cell resolution data
Scanpro is a modular tool for proportion analysis, seamlessly integrating into widely accepted frameworks in the python environment. Scanpro is fast, accurate, support datasets without replicates, and is intended to be used by bioinformatics experts and beginners.

<img src="docs/source/figures/scanpro_workflow.png" width=75% height=55%>

## Install
### PyPi

`pip install scanpro`

### From source: 
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

out = scanpro(adata, clusters_col='clusters', conds_col='condition', samples_col='sample')

```

- If samples_col is not given or set to None, the dataset is assumed to be not replicated and scanpro will run the bootstrapping method.

To show the results, run
```out.results```. 

You can plot the results by calling ```out.plot()```.
