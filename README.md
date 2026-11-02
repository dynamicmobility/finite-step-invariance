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

Choose the sample count, convergence threshold, set type, and system:
```bash
python IterativeAlgorithmMain.py --N 2000 --eps-thresh 0.07 --set-type rbf --system DogBone
```
Set types: `ellipsoid`, `rbf`. Systems (case insensitive): `ExpanderContractor`,
`DogBone`, `CompassGaitWalker`. Defaults are 2000 samples, threshold 0.0,
`ellipsoid`, and `ExpanderContractor`. Thresholds below the attainable
statistical bound for N are raised to that bound with a warning.
