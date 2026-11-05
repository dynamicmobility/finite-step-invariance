"""Run independent seeded trials and save termination and verification metrics."""

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import os
from pathlib import Path

import pandas as pd

from IterativeAlgorithmMain import build_parser, main as run_trial, validate_args


def failed_trial(trial, args, exc):
    return {
        'trial': trial, 'seed': args.seed, 'system': args.system,
        'set_type': args.set_type, 'N': args.N, 'eps_thresh': args.eps_thresh,
        'effective_eps_thresh': None, 'iterations': None, 'converged': False,
        'termination_reason': 'error', 'verification_samples': args.n_verify,
        'verification_escapes': None, 'verification_error_rate': None,
        'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
    }


def execute_trial(trial, args):
    """Module-level worker entry point so it can be pickled by spawn."""
    try:
        if args.verbose:
            result = run_trial(args)
        else:
            with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                result = run_trial(args)
        return {'trial': trial, **result, 'status': 'ok', 'error': ''}
    except Exception as exc:
        return failed_trial(trial, args, exc)


@contextlib.contextmanager
def worker_environment():
    """Spawn workers with single-threaded BLAS and a noninteractive backend."""
    settings = dict.fromkeys(
        ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'], '1')
    settings['MPLBACKEND'] = 'Agg'
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seeds', type=int, nargs='+', help='Explicit seeds to run')
    group.add_argument('--trials', type=int,
                       help='Number of consecutive seeds starting at --seed (default: 10)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes (default: 1, sequential)')
    parser.add_argument('--output', type=Path, help='Summary CSV path')
    parser.add_argument('--verbose', action='store_true', help='Show each trial\'s full output')
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.trials is not None and args.trials <= 0:
        parser.error('--trials must be positive')
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error('--seeds must be nonnegative')
    if args.workers <= 0:
        parser.error('--workers must be positive')
    start = args.seed if args.seed is not None else 0
    seeds = args.seeds if args.seeds is not None else range(start, start + (args.trials or 10))
    output = args.output or Path(
        f'Data/{args.system}_{args.set_type}_seed_sweep_{datetime.now():%Y%m%d_%H%M%S_%f}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    trials = []
    for trial, seed in enumerate(seeds, start=1):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.seed = seed
        trial_args.no_plot = True
        trials.append((trial, trial_args))

    def record(result):
        print(f"Trial {result['trial']}/{len(seeds)}, seed={result['seed']}:", flush=True)
        if result['status'] == 'ok':
            print(f"  {result['iterations']} iterations; {result['termination_reason']}; "
                  f"eps_thresh={result['effective_eps_thresh']}; verification error={result['verification_error_rate']:.4%}", flush=True)
        else:
            print(f"  Failed: {result['error']}", flush=True)
        records.append(result)
        # Only the parent writes; retain input order despite completion order.
        results = pd.DataFrame(records).sort_values('trial').reset_index(drop=True)
        results.to_csv(output, index=False)

    if args.workers == 1:
        for trial, trial_args in trials:
            record(execute_trial(trial, trial_args))
    else:
        with worker_environment(), ProcessPoolExecutor(
                max_workers=min(args.workers, len(trials)),
                mp_context=mp.get_context('spawn')) as executor:
            futures = {executor.submit(execute_trial, trial, trial_args): (trial, trial_args)
                       for trial, trial_args in trials}
            for future in as_completed(futures):
                trial, trial_args = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = failed_trial(trial, trial_args, exc)
                record(result)

    results = pd.DataFrame(records).sort_values('trial').reset_index(drop=True)
    print(f'\nSaved {len(results)} trial results to {output}')
    return results


if __name__ == '__main__':
    main()
