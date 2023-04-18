# Pypropeller: A python implementation of the propeller tool for differnetial proportions analysis
Pypropeller offers a python implementation of the propeller method [(Phipson et al., 2022)](https://academic.oup.com/bioinformatics/article/38/20/4720/6675456) to test the significance of changes in cell proportions
across differnet conditions. It also solves the limitation of needing replicated datasets by simulating artificial replicates using bootstrapping. 

## Install
To install pypropeller: 
- clone the repository
```
git clone https://gitlab.gwdg.de/loosolab/software/pypropeller.git
```
- navigate to pypropeller directory
```
cd pypropeller
```
- then run 
```
pip install .
```
## Quick start
To run the tool import and call the function `pypropeller`:
```
from pypropeller import pypropeller

out = pypropeller.pypropeller(adata, clusters_col='clusters', conds_col='condition', samples_col='sample')

```

- If samples_col is not given or set to None, the dataset is assumed to be not replicated and pypropeller will run the bootstrapping method.

To show the results, run
```out.results```. 

You can plot the results by calling ```out.plot()```.
