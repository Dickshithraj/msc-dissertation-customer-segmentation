"""Run the three new validation analyses added for the dissertation:

1. CLV temporal (calibration/holdout) validation,
2. ROI sensitivity analysis,
3. churn label-threshold sensitivity.

Usage:  python scripts/run_new_validations.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

from src.clv_validation import run_clv_holdout_validation
from src.roi_sensitivity import run_roi_sensitivity
from src.churn_sensitivity import run_churn_threshold_sensitivity


def main() -> None:
    print("=" * 70)
    print("1/3  CLV calibration/holdout validation")
    print("=" * 70)
    print(run_clv_holdout_validation().to_string())

    print("=" * 70)
    print("2/3  ROI sensitivity analysis")
    print("=" * 70)
    print(run_roi_sensitivity().to_string(index=False))

    print("=" * 70)
    print("3/3  Churn label-threshold sensitivity")
    print("=" * 70)
    print(run_churn_threshold_sensitivity().to_string(index=False))


if __name__ == "__main__":
    main()
