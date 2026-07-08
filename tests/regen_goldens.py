"""Regenerate the golden envelope fixtures.

Run after an INTENTIONAL behavior change to the pipeline or the draft()
envelope, then review the golden diff in the PR like any other code
change:

    python3 tests/regen_goldens.py

Requires langgraph (the draft() golden runs the real graph). Offline
mode is forced so the goldens never depend on credentials.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VOICE_OS_OFFLINE"] = "1"

from tests import golden_utils  # noqa: E402


def main() -> int:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        print(
            "regen_goldens: langgraph is required (the draft() golden runs "
            "the real graph): pip install langgraph langgraph-checkpoint-sqlite",
            file=sys.stderr,
        )
        return 1

    envelope = golden_utils.normalize_run_pipeline(
        golden_utils.build_run_pipeline_envelope()
    )
    golden_utils.write_golden(golden_utils.RUN_PIPELINE_GOLDEN, envelope)
    print(f"wrote {golden_utils.RUN_PIPELINE_GOLDEN}")

    with tempfile.TemporaryDirectory(prefix="regen-goldens-") as work_dir:
        envelope = golden_utils.normalize_draft_envelope(
            golden_utils.build_draft_envelope(work_dir)
        )
    golden_utils.write_golden(golden_utils.DRAFT_GOLDEN, envelope)
    print(f"wrote {golden_utils.DRAFT_GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
