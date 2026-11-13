"""Development-only fault stress-search on the authoritative CPU plant (Issue #93).

Subcommands:

  search   start one run (one method, one declared seed) in a new out/ directory
  resume   continue an interrupted run without resetting budgets
  replay   re-run a manifest's selected cases and compare with reported metrics
  compare  matched-budget report over both methods and every declared seed
  run-all  every declared seed for both methods, then the comparison
           (add --resume to continue an interrupted batch)

Exit codes: 0 success; 1 replay mismatch; 2 refused input or output; 3 search
stopped before a budget ended it (continue with resume); 4 replay found no
selected cases to reproduce.

Outputs are biased development discoveries selected with privileged simulator
truth. They are not IID samples, not a validation or test population and not
proof of generalisation. All output is write-once under the ignored out/
directory. Ctrl-C stops a search between plant steps; `resume` continues it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

from aeolus.habitat_v2.stress_search import (
    COMPARISON_FILE_NAME,
    COMPARISON_REPORT_FILE_NAME,
    MANIFEST_FILE_NAME,
    METHODS,
    StressSearchError,
    canonical_bytes,
    collect_runtime_environment,
    compare_runs,
    load_run_manifest,
    load_spec_file,
    render_comparison_markdown,
    replay_manifest,
    resume_search,
    run_search,
    write_once,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(__file__).resolve().with_name("stress_search_spec_v1.json")


class StressSearchCliError(RuntimeError):
    """Command-line misuse or an output location outside out/."""


def _under_out(path: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise StressSearchCliError(
            "stress-search paths must live under the ignored out/ directory"
        ) from error
    if must_exist and not resolved.exists():
        raise StressSearchCliError(f"{path} does not exist")
    if not must_exist and resolved.exists():
        raise StressSearchCliError(f"refusing to overwrite existing output {path}")
    return resolved


class _Cancellation:
    """Turns the first Ctrl-C into a clean stop between plant steps."""

    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, _signum: int, _frame: object) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("stop requested; finishing the current plant step", file=sys.stderr)

    def __call__(self) -> bool:
        return self.requested


def _report(summary: object) -> None:
    print(
        f"{summary.method} seed={summary.seed}: {summary.end_reason}; "
        f"{summary.simulation_calls_used} simulation calls, "
        f"{summary.candidate_attempts_used} attempts; "
        + (
            f"manifest {summary.manifest_sha256}"
            if summary.final
            else "not final, continue with `resume`"
        ),
        file=sys.stderr,
    )


def _search(arguments: argparse.Namespace) -> int:
    output = _under_out(arguments.output, must_exist=False)
    cancellation = _Cancellation()
    cancellation.install()
    summary = run_search(
        spec=load_spec_file(arguments.spec),
        repo_root=REPO_ROOT,
        output_dir=output,
        method=arguments.method,
        seed=arguments.seed,
        cancel_requested=cancellation,
        runtime_environment=collect_runtime_environment(REPO_ROOT),
    )
    _report(summary)
    return 0 if summary.final else 3


def _resume(arguments: argparse.Namespace) -> int:
    output = _under_out(arguments.run, must_exist=True)
    cancellation = _Cancellation()
    cancellation.install()
    summary = resume_search(
        output_dir=output,
        repo_root=REPO_ROOT,
        cancel_requested=cancellation,
        runtime_environment=collect_runtime_environment(REPO_ROOT),
    )
    _report(summary)
    return 0 if summary.final else 3


def _replay(arguments: argparse.Namespace) -> int:
    manifest = _under_out(arguments.manifest, must_exist=True)
    output = _under_out(arguments.output, must_exist=False)
    receipt = replay_manifest(
        manifest_path=manifest,
        repo_root=REPO_ROOT,
        output_dir=output,
        runtime_environment=collect_runtime_environment(REPO_ROOT),
    )
    print(
        f"replay {receipt['status']}: {receipt['reproduced_count']} of "
        f"{receipt['case_count']} selected cases reproduced within "
        f"{receipt['metric_absolute_tolerance']}",
        file=sys.stderr,
    )
    if receipt["status"] == "PASS":
        return 0
    # NO_CASES: nothing was selected, so nothing was reproduced; that is not a pass.
    return 4 if receipt["status"] == "NO_CASES" else 1


def _write_comparison(run_directories: list[Path], output: Path) -> None:
    manifests = [load_run_manifest(directory / MANIFEST_FILE_NAME) for directory in run_directories]
    comparison = compare_runs(manifests)
    # write_once accepts identical content, so re-running a finished batch is harmless.
    output.mkdir(parents=True, exist_ok=True)
    write_once(output / COMPARISON_FILE_NAME, canonical_bytes(comparison))
    write_once(
        output / COMPARISON_REPORT_FILE_NAME,
        render_comparison_markdown(comparison).encode("utf-8"),
    )
    print(f"comparison {comparison['comparison_sha256']}", file=sys.stderr)


def _compare(arguments: argparse.Namespace) -> int:
    runs = [_under_out(path, must_exist=True) for path in arguments.runs]
    _write_comparison(runs, _under_out(arguments.output, must_exist=False))
    return 0


def _run_all(arguments: argparse.Namespace) -> int:
    output = _under_out(arguments.output, must_exist=arguments.resume)
    spec = load_spec_file(arguments.spec)
    # Check every per-run directory before anything is written.
    directories = {
        (method, seed): _under_out(
            output / f"{method}-{seed}", must_exist=False
        ) if not arguments.resume else (output / f"{method}-{seed}").resolve()
        for method in METHODS
        for seed in spec.seeds
    }
    spec_bytes = canonical_bytes(spec.to_mapping())
    for directory in directories.values():
        if directory.parent != output:
            raise StressSearchCliError("per-run directory escapes the batch directory")
        if not directory.exists():
            continue
        # An existing per-run directory must be a run of this same spec.
        stored = directory / "spec.json"
        if not stored.is_file() or not (directory / "audit.jsonl").is_file():
            raise StressSearchCliError(
                f"{directory.name} exists but is not a started run; remove the batch and start again"
            )
        if stored.read_bytes() != spec_bytes:
            raise StressSearchCliError(
                f"{directory.name} was produced by a different spec than the one given"
            )
    cancellation = _Cancellation()
    cancellation.install()
    environment = collect_runtime_environment(REPO_ROOT)
    for (method, seed), directory in directories.items():
        if (directory / MANIFEST_FILE_NAME).exists():
            print(f"{method} seed={seed}: already complete", file=sys.stderr)
            continue
        if directory.exists():
            summary = resume_search(
                output_dir=directory,
                repo_root=REPO_ROOT,
                cancel_requested=cancellation,
                runtime_environment=environment,
            )
        else:
            summary = run_search(
                spec=spec,
                repo_root=REPO_ROOT,
                output_dir=directory,
                method=method,
                seed=seed,
                cancel_requested=cancellation,
                runtime_environment=environment,
            )
        _report(summary)
        if not summary.final:
            print(
                "batch stopped; continue it with the same command plus --resume",
                file=sys.stderr,
            )
            return 3
    _write_comparison(list(directories.values()), output / "comparison")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="start one search run")
    search.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    search.add_argument("--method", choices=METHODS, required=True)
    search.add_argument("--seed", required=True, help="one of the seeds declared in the spec")
    search.add_argument("--output", type=Path, required=True, help="new directory under out/")
    search.set_defaults(handler=_search)

    resume = commands.add_parser("resume", help="continue an interrupted run")
    resume.add_argument("--run", type=Path, required=True, help="existing run directory")
    resume.set_defaults(handler=_resume)

    replay = commands.add_parser("replay", help="replay a manifest's selected cases")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True, help="new directory under out/")
    replay.set_defaults(handler=_replay)

    compare = commands.add_parser("compare", help="matched-budget comparison report")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True, help="new directory under out/")
    compare.set_defaults(handler=_compare)

    run_all = commands.add_parser("run-all", help="all declared seeds, both methods, then compare")
    run_all.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    run_all.add_argument("--output", type=Path, required=True, help="new directory under out/")
    run_all.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted batch in an existing --output directory",
    )
    run_all.set_defaults(handler=_run_all)

    arguments = parser.parse_args(argv)
    try:
        return arguments.handler(arguments)
    except (StressSearchError, StressSearchCliError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: {type(error).__name__}: {error.strerror or error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
