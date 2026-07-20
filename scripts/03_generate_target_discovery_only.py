#!/usr/bin/env python3
"""
Fast script to generate TARGET discovery rollouts only.
Skips BASE (already done), uses frozen manifest.
Run with: python scripts/03_generate_target_discovery_only.py --config <config.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit.commands import stage03_main

if __name__ == "__main__":
    # Parse config from command line, then call stage03_main with TARGET condition
    parser = argparse.ArgumentParser(description="Generate TARGET discovery rollouts only")
    parser.add_argument("--config", required=True, help="Audit config YAML file")
    args, remaining = parser.parse_known_args()
    
    # Call stage03_main with --config and --condition TARGET
    stage03_main(["--config", args.config, "--condition", "TARGET"] + remaining)
