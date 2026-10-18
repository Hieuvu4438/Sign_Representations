"""Shared CLI plumbing: --config/--set parsing, error handling with non-zero exit codes."""
from __future__ import annotations

import argparse
import sys
import traceback

from asm_jepa.config import ConfigError, load_config


def add_config_args(p: argparse.ArgumentParser, default=None, required=False):
    p.add_argument("--config", default=default, required=required, help="YAML config (supports `inherits`)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override an existing config key (YAML-parsed value); repeatable")


def config_from_args(args, schema: str):
    return load_config(args.config, schema, args.set)


def run(main):
    """Execute `main()`; map expected errors to exit code 2, unexpected ones to 1."""
    try:
        code = main()
        sys.exit(code or 0)
    except (ConfigError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        sys.exit(2)
