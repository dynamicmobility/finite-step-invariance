"""Run independent seeded trials and save termination and verification metrics."""

import argparse
import contextlib
from datetime import datetime
import os
from pathlib import Path

import pandas as pd

from IterativeAlgorithmMain import build_parser, main as run_trial, validate_args


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seeds', type=int, nargs='+', help='Explicit seeds to run')
    group.add_argument('--trials', type=int,
                       help='Number of consecutive seeds starting at --seed (default: 10)')
    parser.add_argument('--output', type=Path, help='Summary CSV path')
    parser.add_argument('--verbose', action='store_true', help='Show each trial\'s full output')
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.trials is not None and args.trials <= 0:
        parser.error('--trials must be positive')
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error('--seeds must be nonnegative')
    start = args.seed if args.seed is not None else 0
    seeds = args.seeds if args.seeds is not None else range(start, start + (args.trials or 10))
    output = args.output or Path(
        f'Data/{args.system}_{args.set_type}_seed_sweep_{datetime.now():%Y%m%d_%H%M%S_%f}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for trial, seed in enumerate(seeds, start=1):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.seed = seed
        trial_args.no_plot = True
        print(f'Trial {trial}/{len(seeds)}, seed={seed}:', flush=True)
        try:
            if args.verbose:
                result = run_trial(trial_args)
            else:
                with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                    result = run_trial(trial_args)
            result = {'trial': trial, **result, 'status': 'ok', 'error': ''}
            print(f"  {result['iterations']} iterations; {result['termination_reason']}; "
                  f"verification error={result['verification_error_rate']:.4%}", flush=True)
        except Exception as exc:
            result = {
                'trial': trial, 'seed': seed, 'system': args.system,
                'set_type': args.set_type, 'N': args.N, 'eps_thresh': args.eps_thresh,
                'effective_eps_thresh': None, 'iterations': None, 'converged': False,
                'termination_reason': 'error', 'verification_samples': args.n_verify,
                'verification_escapes': None, 'verification_error_rate': None,
                'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
            }
            print(f"  Failed: {result['error']}", flush=True)
        records.append(result)
        # Preserve completed trials even if a later trial is interrupted.
        pd.DataFrame(records).to_csv(output, index=False)

    results = pd.DataFrame(records)
    print(f'\nSaved {len(results)} trial results to {output}')
    return results


if __name__ == '__main__':
    main()
