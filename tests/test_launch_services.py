from pathlib import Path


def test_wda_start_disables_optional_xctest_failure_diagnostics():
    """WDA startup must not block on a diagnostic archive before runner launch."""
    source = Path("scripts/launch_services.py").read_text()
    assert '"-collect-test-diagnostics", "never"' in source
