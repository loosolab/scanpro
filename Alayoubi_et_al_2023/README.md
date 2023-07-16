## Create conda environment
```$ mamba create -n scanpro_analysis python=3.9 r-base=4.3 r-devtools r-seurat ipykernel ```   
```$ conda activate scanpro_analysis ```

## Install scanpro
`$ pip install .`

## Install sccoda
`$ pip install sccoda `

## Install propeller from the speckle package
`$ pip install rpy2`

In R, run:
```R
library(devtools)
devtools::install_github("phipsonlab/speckle")
```

## Register the kernel for jupyter notebooks
`$ python -m ipykernel install --user --name scanpro_analysis `
