#!/usr/bin/env python3
"""python -m cropai.validate  /  python -m cropai validate-final"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cropai.validate.report import final_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CROP-AI Phase 6 final validation harness")
    parser.add_argument("--smoke-image", default="", help="optional image for untrained e2e smoke")
    parser.add_argument("--crop", default="rice")
    args = parser.parse_args(argv)
    report = final_report()
    if args.smoke_image:
        from cropai.validate.e2e import smoke_e2e

        report["e2e_smoke"] = smoke_e2e(Path(args.smoke_image), crop=args.crop)
    print(json.dumps(report, indent=2))
    return 0 if report.get("project_complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
