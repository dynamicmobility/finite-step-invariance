"""Print epsilon_star and final convergence epsilon statistics for seed sweeps."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "Final_Data",
        help="Directory containing sweep CSVs (default: Final_Data beside this script)",
    )
    args = parser.parse_args()
    sweeps = sorted(args.data.rglob("*seed_sweep*.csv"))
    if not sweeps:
        parser.error(f"No seed sweep CSVs found in {args.data}")

    headers = ("Sweep", "Trials", "effective_eps_thresh", "Mean epsilon_star", "Std epsilon_star",
               "Mean last epsilon", "Std last epsilon", "Mean iterations")
    report = []
    for path in sweeps:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            parser.error(f"Sweep CSV is empty: {path}")
        try:
            thresholds = sorted({float(row["effective_eps_thresh"]) for row in rows})
            values = [float(row["epsilon_star"]) for row in rows]
            iterations = [int(row["iterations"]) for row in rows]
            last_epsilons = []
            for row in rows:
                logs = sorted(path.parent.glob(f"*_convergence_log_*_seed{row['seed']}.csv"))
                if len(logs) != 1:
                    parser.error(
                        f"Expected one convergence log for seed {row['seed']} in "
                        f"{path.parent}, found {len(logs)}"
                    )
                with logs[0].open(newline="") as stream:
                    convergence = list(csv.DictReader(stream))
                if not convergence:
                    parser.error(f"Convergence log is empty: {logs[0]}")
                last_epsilons.append(float(convergence[-1]["epsilon"]))
        except (KeyError, ValueError) as error:
            parser.error(f"Invalid sweep data in {path}: {error}")
        report.append((
            str(path.relative_to(args.data)),
            str(len(values)),
            ", ".join(f"{value:.8f}" for value in thresholds),
            f"{mean(values):.8f}",
            f"{stdev(values):.8f}" if len(values) > 1 else "N/A",
            f"{mean(last_epsilons):.8f}",
            f"{stdev(last_epsilons):.8f}" if len(last_epsilons) > 1 else "N/A",
            f"{mean(iterations):.2f}",
        ))

    widths = [max(len(row[i]) for row in [headers, *report]) for i in range(len(headers))]
    for row in [headers, *report]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    print("\nStandard deviations use the sample formula (ddof=1).")


if __name__ == "__main__":
    main()
