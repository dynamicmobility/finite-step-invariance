# finite-step-invariance

## Setup

### Create the conda environment

```bash
conda env create -f environment.yaml
conda activate finite-invariance
```

### Update an existing environment

```bash
conda env update -f environment.yaml --prune
```

## Scripts

1. Main plotting script: The following script will recreate the main plots that are included in the manuscript using the pre-obtained algorithm data stored in `Data`.
```bash
bash CDC_make_plots.sh
```

2. Main algorithm script:
```bash
python IterativeAlgorithmMain.py
```
