"""Package entry point: python3 -m voice_os <command>.

Delegates to the product-layer CLI (voice_os/product/cli.py). Importing
this module stays stdlib-only; the langgraph dependency loads only when
a graph-backed command actually runs.
"""

from __future__ import annotations

import sys

from .product.cli import main

if __name__ == "__main__":
    sys.exit(main())
