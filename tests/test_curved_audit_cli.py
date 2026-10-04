import importlib.util
import json
from pathlib import Path
import sys
import pytest
from trueskate_ai.research.curved_audit import manifest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/collection/run_curved_audit.py'
spec=importlib.util.spec_from_file_location('curved_cli',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_explicit_schedule_matches_freeze_and_rejects_execution_mismatch(tmp_path,monkeypatch):
    path=tmp_path/'manifest.json'
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--freeze','--schedule','v3','--manifest',str(path)])
    runner.main()
    assert json.loads(path.read_text())==json.loads(json.dumps(manifest('v3')))
    monkeypatch.setattr(runner.socket,'gethostname',lambda:'training-server')
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--schedule','v2','--manifest',str(path),
                        '--out',str(tmp_path/'run'),'--wda-revision','fixture'])
    with pytest.raises(SystemExit) as exc:runner.main()
    assert exc.value.code==2 and not (tmp_path/'run').exists()
