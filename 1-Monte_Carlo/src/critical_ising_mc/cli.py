from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .crops import create_fixed_crop_manifests
from .diagnostics import diagnose_configs, diagnose_split
from .fixed_background import run_fixed_background_check
from .parent_size import run_parent_size_check
from .sampling import generate_split
from .verification import verify_configs, verify_crop_manifest

PRODUCTION_NAMES = (
    "critical_L512_train.yaml",
    "critical_L512_val.yaml",
    "critical_L512_reference_a.yaml",
    "critical_L512_reference_b.yaml",
)


def production_configs(config_dir: str | Path) -> list[Path]:
    directory = Path(config_dir).resolve()
    paths = [directory / name for name in PRODUCTION_NAMES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing production configs: {', '.join(missing)}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ising-mc", description="Critical Ising Wolff data pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate or resume one split")
    generate.add_argument("--config", required=True)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--force", action="store_true")

    generate_all = subparsers.add_parser(
        "generate-all", help="Generate all four production splits"
    )
    generate_all.add_argument("--config-dir", default="configs")
    generate_all.add_argument("--workers", type=int, default=1)
    generate_all.add_argument("--force", action="store_true")

    diagnose = subparsers.add_parser(
        "diagnose", help="Run physical and numerical acceptance checks"
    )
    diagnose.add_argument("--config-dir", default="configs")

    crops = subparsers.add_parser(
        "make-crops", help="Create frozen validation/reference crop manifests"
    )
    crops.add_argument("--config-dir", default="configs")
    crops.add_argument("--spec", default="configs/crops.yaml")

    verify = subparsers.add_parser(
        "verify", help="Verify shards, checksums, IDs, seeds, and crops"
    )
    verify.add_argument("--config-dir", default="configs")
    verify.add_argument("--skip-checksums", action="store_true")

    smoke = subparsers.add_parser(
        "smoke", help="Run the non-production L=32 smoke pipeline"
    )
    smoke.add_argument("--config", default="configs/smoke_L32.yaml")
    smoke.add_argument("--workers", type=int, default=2)
    smoke.add_argument("--force", action="store_true")

    size_check = subparsers.add_parser(
        "parent-size-check", help="Compare L=512 and L=1024 crop references"
    )
    size_check.add_argument("--config-dir", default="configs")
    size_check.add_argument("--spec", default="configs/crops.yaml")
    size_check.add_argument("--max-distance", type=int, default=64)

    fixed_background = subparsers.add_parser(
        "fixed-background-check",
        help="Evaluate training crops within the fixed L=512 background",
    )
    fixed_background.add_argument("--config-dir", default="configs")
    fixed_background.add_argument("--spec", default="configs/crops.yaml")
    fixed_background.add_argument("--distance-fraction", type=float, default=0.25)

    run_all = subparsers.add_parser(
        "run-all", help="Generate, diagnose, verify, and freeze crops"
    )
    run_all.add_argument("--config-dir", default="configs")
    run_all.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_split(args.config, args.workers, args.force)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "generate-all":
            result = {}
            for path in production_configs(args.config_dir):
                result[path.stem] = generate_split(path, args.workers, args.force)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "diagnose":
            result = diagnose_configs(production_configs(args.config_dir))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["passed"] else 2
        elif args.command == "make-crops":
            paths = create_fixed_crop_manifests(
                production_configs(args.config_dir), args.spec
            )
            print(
                json.dumps({"val": str(paths[0]), "reference": str(paths[1])}, indent=2)
            )
        elif args.command == "verify":
            configs = production_configs(args.config_dir)
            result = verify_configs(configs, not args.skip_checksums)
            manifest_dir = load_config(configs[0]).manifest_dir
            for name in ("val_crops.csv", "reference_crops.csv"):
                path = manifest_dir / name
                if path.exists():
                    result[name] = verify_crop_manifest(
                        path, manifest_dir / "parents.csv"
                    )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "smoke":
            generated = generate_split(args.config, args.workers, args.force)
            config = load_config(args.config)
            verified = verify_configs([args.config], True)
            _, diagnostics, _ = diagnose_split(config)
            smoke_passed = bool(
                verified["passed"]
                and diagnostics["checks"]["array_contract"]
                and diagnostics["checks"]["invariants"]
            )
            result = {
                "passed": smoke_passed,
                "generated": generated,
                "verification": verified,
                "diagnostics": diagnostics,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if smoke_passed else 2
        elif args.command == "parent-size-check":
            result = run_parent_size_check(
                args.config_dir, args.spec, args.max_distance
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["passed"] else 2
        elif args.command == "fixed-background-check":
            result = run_fixed_background_check(
                args.config_dir, args.spec, args.distance_fraction
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["passed"] else 2
        elif args.command == "run-all":
            configs = production_configs(args.config_dir)
            for path in configs:
                generate_split(path, args.workers, False)
            diagnostics = diagnose_configs(configs)
            if not diagnostics["passed"]:
                print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
                return 2
            crop_spec = Path(args.config_dir) / "crops.yaml"
            create_fixed_crop_manifests(configs, crop_spec)
            fixed_background = run_fixed_background_check(args.config_dir, crop_spec)
            result = verify_configs(configs, True)
            print(
                json.dumps(
                    {
                        "diagnostics": diagnostics,
                        "fixed_background": fixed_background,
                        "verification": result,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0 if fixed_background["passed"] else 2
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary converts failures to a clean exit code.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
