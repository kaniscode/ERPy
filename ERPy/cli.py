from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path

import yaml


def _template_path(name: str) -> Path:
    return Path(resources.files("ERPy.templates") / name)


def init_project(config_path: Path, data_root: Path, config_only: bool = False, dry_run: bool = False, force: bool = False) -> list[Path]:
    written: list[Path] = []
    config_path = config_path.expanduser()
    data_root = data_root.expanduser()
    with _template_path("config.example.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.update(
        {
            "rawdata_path": str(data_root / "raw"),
            "procdata_path": str(data_root / "procdata"),
            "rawdata_meta_path": str(data_root / "metadata" / "raw_metadata.csv"),
            "stim_meta_path": str(data_root / "metadata" / "stim_metadata.csv"),
            "elec_meta_path": str(data_root / "metadata" / "electrode_metadata.csv"),
        }
    )
    if not dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists() and not force:
            raise FileExistsError(f"{config_path} exists; pass --force to overwrite")
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    written.append(config_path)
    if config_only:
        return written
    for sub in ("raw", "procdata", "metadata"):
        path = data_root / sub
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)
    examples = {
        "raw_metadata.example.csv": "raw_metadata.csv",
        "stim_metadata.example.csv": "stim_metadata.csv",
        "electrode_metadata.example.csv": "electrode_metadata.csv",
    }
    for src, dst in examples.items():
        out = data_root / "metadata" / dst
        if not dry_run:
            if out.exists() and not force:
                continue
            shutil.copyfile(_template_path(src), out)
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize an ERPy project config and metadata stubs.")
    parser.add_argument("command", nargs="?", default="init", choices=["init", "show-config"])
    parser.add_argument("--config", default=str(Path.home() / ".erpy" / "config.yaml"))
    parser.add_argument("--data-root", default=str(Path.home() / "erpy_data"))
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "show-config":
        from .utils import load_config

        print(yaml.safe_dump(load_config(args.config), sort_keys=False))
        return 0
    written = init_project(
        Path(args.config),
        Path(args.data_root),
        config_only=args.config_only,
        dry_run=args.dry_run,
        force=args.force,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
