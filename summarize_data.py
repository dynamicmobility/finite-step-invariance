"""Summarize several run CSVs (columns: epsilon,volume) into plot-ready CSVs.

Usage:
    python summarize.py run_*.csv --prefix dogbone
Writes:
    dogbone_epsilon_summary.csv   (iter,mean,lower,upper) for 1 - epsilon
    dogbone_volume_summary.csv    (iter,mean,lower,upper) for volume
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats


def summarize(runs, column, transform=lambda x: x, band="ci95"):
    # rows = iterations, columns = runs
    data = np.column_stack([transform(r[column].to_numpy()) for r in runs])
    n = data.shape[1]
    mean = data.mean(axis=1)
    std = data.std(axis=1, ddof=1)
    sem = std / np.sqrt(n)
    if band == "ci95":
        half = stats.t.ppf(0.975, df=n - 1) * sem
    elif band == "sem":
        half = sem
    elif band == "std":
        half = std
    else:
        raise ValueError(band)
    return pd.DataFrame({
        "iter": np.arange(data.shape[0]),
        "mean": mean,
        "lower": mean - half,
        "upper": mean + half,
    })


def nice_step(span):
    """A round axis step (1, 2 or 5 times a power of ten) giving ~5 intervals."""
    raw = max(span, 1e-12) / 5
    mag = 10 ** np.floor(np.log10(raw))
    for k in (1, 2, 5, 10):
        if raw <= k * mag:
            return k * mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--prefix", default="summary")
    ap.add_argument("--band", choices=["ci95", "sem", "std"], default="std",
                    help="half-width of the lower/upper band around the mean")
    ap.add_argument("--pattern", default="*convergence_log*.csv",
                    help="glob used to pick files when a directory is given")
    ap.add_argument("--target", type=float, default=None,
                    help="target accuracy, e.g. 0.97, for the iterations-to-converge stat")
    ap.add_argument("--sweep", default=None,
                    help="seed-sweep summary CSV (epsilon_star, iterations); "
                         "auto-detected as *seed_sweep*.csv when a directory is given")
    ap.add_argument("--iter-start", type=int, default=0,
                    help="value of the first row in the iter column (use 1 if your "
                         "pgfplots x-axis starts at 1)")
    args = ap.parse_args()

    # expand directories into the CSV files inside them
    files = []
    for f in args.files:
        if os.path.isdir(f):
            files += sorted(glob.glob(os.path.join(f, args.pattern)))
        else:
            files.append(f)
    if not files:
        sys.exit("No CSV files found in: " + " ".join(args.files))
    print(f"Reading {len(files)} files:")
    for f in files:
        print("  ", f)

    runs = [pd.read_csv(f) for f in files]

    # ---- numbers for the figure caption (computed on the full, untruncated runs)
    print(f"\nCaption statistics over {len(runs)} runs (mean +/- std):")
    print(f"  run lengths: {[len(r) for r in runs]}")
    final_vol = np.array([r["volume"].iloc[-1] for r in runs])
    print(f"  final volume:         {final_vol.mean():.4f} +/- {final_vol.std(ddof=1):.4f}")

    # epsilon* and iteration counts come from the seed-sweep summary CSV
    sweep_path = args.sweep
    if sweep_path is None:
        for f in args.files:
            if os.path.isdir(f):
                found = sorted(glob.glob(os.path.join(f, "*seed_sweep*.csv")))
                if found:
                    sweep_path = found[0]
                    break
    if sweep_path is not None:
        sw = pd.read_csv(sweep_path)
        if "status" in sw.columns:
            sw = sw[sw["status"] == "ok"]
        eps_star = sw["epsilon_star"].to_numpy()
        acc_star = 1 - eps_star
        iters = sw["iterations"].to_numpy()
        print(f"  (from {os.path.basename(sweep_path)}, {len(sw)} runs)")
        print(f"  epsilon*:             {eps_star.mean():.4f} +/- {eps_star.std(ddof=1):.4f}")
        print(f"  accuracy 1-eps*:      {acc_star.mean():.4f} +/- {acc_star.std(ddof=1):.4f} "
              f"(min {acc_star.min():.4f}, max {acc_star.max():.4f})")
        print(f"  iterations:           {iters.mean():.1f} +/- {iters.std(ddof=1):.1f} "
              f"(min {iters.min()}, max {iters.max()})")
        if "converged" in sw.columns:
            print(f"  converged:            {int(sw['converged'].sum())}/{len(sw)} runs")
    else:
        print("  (no seed-sweep CSV found; pass --sweep to get epsilon* and iteration stats)")
    if args.target is not None and sweep_path is None:
        its = []
        for f, r in zip(files, runs):
            hit = np.flatnonzero((1 - r["epsilon"].to_numpy()) >= args.target)
            if len(hit):
                its.append(hit[0] + 1)  # 1-indexed iteration count
            else:
                print(f"  (target {args.target} never reached in {os.path.basename(f)})")
        if its:
            its = np.array(its)
            print(f"  iterations to reach 1-eps >= {args.target}: "
                  f"{its.mean():.1f} +/- {its.std(ddof=1) if len(its) > 1 else 0:.1f} "
                  f"(min {its.min()}, max {its.max()}, {len(its)}/{len(runs)} runs)")
    print()

    # runs may differ in length; truncate to the shortest
    m = min(len(r) for r in runs)
    runs = [r.iloc[:m] for r in runs]

    summaries = {
        "epsilon": summarize(runs, "epsilon", lambda e: 1 - e, args.band),
        "volume": summarize(runs, "volume", band=args.band),
    }
    for name, df in summaries.items():
        df["iter"] = df["iter"] + args.iter_start
        out = f"{args.prefix}_{name}_summary.csv"
        df.to_csv(out, index=False)
        print("Wrote", os.path.abspath(out))

    # ---- values for the pgfplots axes
    acc, vol = summaries["epsilon"], summaries["volume"]
    xs = acc["iter"].to_numpy()
    print(f"\nTikZ info (band = {args.band}, {len(xs)} plotted iterations, "
          f"iter column starts at {xs[0]}):")
    print(f"  xmin={xs[0]}, xmax={xs[-1]}, xtick={{{', '.join(str(x) for x in xs)}}}")
    for label, df, step in [("accuracy 1-eps (left axis)", acc, 0.05),
                            ("volume (right axis)", vol, None)]:
        lo, hi = df["lower"].min(), df["upper"].max()
        if step is None:
            step = nice_step(hi - lo)
        print(f"  {label}: data range [{lo:.4f}, {hi:.4f}]  ->  "
              f"ymin={np.floor(lo / step) * step:.4g}, ymax={np.ceil(hi / step) * step:.4g}")
    if args.target is not None:
        print(f"  target accuracy line: y={args.target}")
    print(f"  mean at last iteration: 1-eps={acc['mean'].iloc[-1]:.4f}, "
          f"volume={vol['mean'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()