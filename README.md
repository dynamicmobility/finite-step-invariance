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
Set types: `ell`, `rbf`. Systems (case insensitive): `cec`,
`nec`, `cgw`. Defaults are 2000 samples, threshold 0.0,
`ell`, and `cec`. Thresholds below the attainable
statistical bound for N are raised to that bound with a warning.

### Seed sweeps

Run ten independent trials with seeds 0 through 9:
```bash
python SweepSeeds.py --trials 10 --seed 0 --N 2000 --eps-thresh 0.07 --set-type ell --system cec
```
Or select seeds explicitly:
```bash
python SweepSeeds.py --seeds 3 17 42 --set-type rbf --system nec --output Data/seed_sweep.csv
```
The sweep skips plots, prints a result per trial, and writes a summary CSV after
each trial. It reports `iterations` (number of iterations executed, starting at
1), `converged`, `termination_reason`, and the post-hoc `verification_error_rate`
(escaped points divided by verification samples). Trials reaching the iteration
limit are distinguished from convergence; failed trials are recorded and the
sweep continues. `--max-iter` defaults to 200 and `--n-verify` to 5000.
Use `--verbose` to display full trial output. Without `--output`, the CSV goes to
`Data/` with the system, set type, and timestamp in its filename.

Run trials in parallel using spawned worker processes:
```bash
python SweepSeeds.py --trials 10 --seed 0 --workers 4 --system nec --set-type ell
```
`--workers 1` (the default) runs sequentially. Parallel workers use a
noninteractive plotting backend and limit BLAS/OpenMP threads to one each.
JAX and solver runtimes may still use additional threads. Results are printed
as trials finish; the summary CSV stays sorted by the original trial order.
Only the parent process writes the summary CSV.
