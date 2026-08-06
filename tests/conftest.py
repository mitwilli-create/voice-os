import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _default_api_only_billing(monkeypatch):
    """Tests must never shell out to a real subscription CLI.

    VOICE_OS_SUBSCRIPTION_FIRST defaults on in production so live drafts
    prefer a paid CLI subscription over a metered API key. Unit tests
    exercise the metered adapters directly and must stay hermetic, so this
    autouse fixture forces the kill switch off for every test unless a
    test explicitly re-enables it (and mocks the subscription_cli
    functions) to test subscription-routing behavior itself.
    """
    monkeypatch.setenv("VOICE_OS_SUBSCRIPTION_FIRST", "0")
