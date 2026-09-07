import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("archive_check", Path(__file__).parents[1] / "scripts/ops/check_archive.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def entry(number=1):
    return (f"## ARCH-{number:03d} — Example\n"
            "- Description: Retired implementation.\n"
            f"- Commit: `{'a' * 40}`\n"
            "- Tag: `archive/example`\n"
            f"- Paths: [source](https://github.com/AsherNoble/TrueSkate-AI/tree/{'a' * 40}/src)\n"
            "- Recovery: `git show archive/example:src/example.py`\n"
            "- Artifacts: None.\n").encode()


def test_unchanged_and_append_and_bootstrap():
    old = b"# Archive\n\n" + entry()
    for base, proposed in [(b"", old), (old, old), (old, old + b"\n" + entry(2))]:
        check.validate_archive(base, proposed)


@pytest.mark.parametrize("transform", [lambda s:b"", lambda s:s[:-1], lambda s:s.replace(b"Retired", b"Changed")])
def test_existing_content_cannot_be_rewritten(transform):
    with pytest.raises(ValueError, match="append-only"):
        check.validate_archive(entry(), transform(entry()))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        check.validate_archive(entry(), entry() + entry())


@pytest.mark.parametrize("bad", [entry().replace(b"- Recovery:", b"Recovery:"), entry().replace(b"a" * 40, b"abcd")])
def test_incomplete_provenance_rejected(bad):
    with pytest.raises(ValueError):
        check.validate_archive(b"", bad)
