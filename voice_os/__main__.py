"""Package entry point: python3 -m voice_os <command>.

Delegates to the product-layer CLI (voice_os/product/cli.py). Importing
this module stays stdlib-only; the langgraph dependency loads only when
a graph-backed command actually runs.
"""

from __future__ import annotations

import sys

# The product-layer state schema (voice_os/product/graph.py: VoiceState)
# uses PEP 604 `X | None` unions, which langgraph evaluates via
# get_type_hints() at graph-build time. That evaluation raises an opaque
# `unsupported operand type(s) for |` TypeError on Python < 3.10. Fail
# fast here with an actionable message instead. A common trap: a shell
# where `python3` resolves to system Python 3.9 (e.g. non-interactive
# shells without the pyenv shim on PATH) while the project default is
# the project contract is `python3` — ensure that command points to Python 3.14.
if sys.version_info < (3, 10):
    _v = ".".join(str(n) for n in sys.version_info[:3])
    sys.stderr.write(
        f"voice_os requires Python 3.10+ (PEP 604 union type hints); "
        f"this interpreter is {_v} at {sys.executable}. "
        f"Re-run with the project interpreter, e.g. `python3 -m voice_os ...`.\n"
    )
    sys.exit(2)

from .product.cli import main  # noqa: E402  (imported after the version guard)

if __name__ == "__main__":
    sys.exit(main())
