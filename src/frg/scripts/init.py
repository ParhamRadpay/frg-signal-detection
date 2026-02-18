"""
Initialization script for the frg package.

This script provides a CLI command to initialize a user workspace by copying template configuration files from the package distribution to the current working directory.

Author: Riccardo Finotello <riccardo.finotello@cea.fr>
"""

import argparse
import importlib.resources as pkg_resources
import shutil
from pathlib import Path


def copy_resource_dir(resource_pkg: str, target_dir: Path):
    """
    Copy a directory from package resources to a target path.
    """
    try:
        # Use importlib.resources to find the path to the internal data
        with pkg_resources.path(resource_pkg, "__init__.py") as p:
            source_dir = p.parent

        if not source_dir.exists():
            print(f"[ERROR] Source directory {source_dir} not found.")
            return

        print(f"[INFO] Initializing {target_dir.name} from {source_dir}...")

        # Copy files (avoiding recursive directory copying if simple)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.name == "__init__.py" or item.name == "__pycache__":
                continue

            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    except Exception as e:
        print(f"[ERROR] Failed to copy resources: {e}")


def main(argv: list[str] | None = None) -> int | str:
    parser = argparse.ArgumentParser(
        description="Initialize the frg workspace by copying template configs and scripts."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files."
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()

    # 1. Copy configs
    configs_dir = cwd / "configs"
    if configs_dir.exists() and args.force:
        shutil.rmtree(configs_dir)
    copy_resource_dir("frg.configs", configs_dir)

    # 2. Copy env.sh specifically
    env_file = cwd / "env.sh"
    if env_file.exists() and not args.force:
        print("[INFO] env.sh already exists. Use --force to overwrite.")
    else:
        try:
            with pkg_resources.path("frg.scripts", "env.sh") as p:
                if p.exists():
                    print(f"[INFO] Initializing env.sh from {p}...")
                    shutil.copy2(p, env_file)
        except Exception as e:
            print(f"[WARN] Could not copy env.sh: {e}")

    # 3. Copy notebooks
    notebooks_dir = cwd / "notebooks"
    if notebooks_dir.exists() and args.force:
        shutil.rmtree(notebooks_dir)
    copy_resource_dir("frg.notebooks", notebooks_dir)

    print(
        "\n[SUCCESS] Workspace initialized. You can now customize your configs and run the simulations."
    )

    return 0


def cli():
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
