"""Produce an inconclusive/pass stage gate and grouped spacing evidence."""
import argparse
import json
from pathlib import Path
from trueskate_ai.research.curve_protocol import verify_manifest,save_new,digest,command_manifest
from trueskate_ai.research.curve_report import stage_decision,grouped_summary,reviewed_rows,rule_results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--stage',choices=['pilot','main','confirmation'],required=True)
    p.add_argument('--measurements',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--annotations',type=Path)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());verify_manifest(manifest,check_source=False)
    rows=[r for path in a.measurements for r in json.loads(path.read_text())['rows']]
    marks=json.loads(a.annotations.read_text()) if a.annotations else {}
    if marks and marks.get('rows_sha256')!=digest(rows):p.error('annotation order/measurement identity mismatch')
    parse=lambda name:{int(k):v for k,v in marks.get(name,{}).items()}
    gate=json.loads(a.gate.read_text()) if a.gate else None
    automated_evaluable=sum(r['metrics']['evaluable'] for r in rows)
    rows=reviewed_rows(rows,manifest,parse('annotations'))
    result=stage_decision(manifest,rows,stage=a.stage,pilot_gate=gate,annotations=parse('annotations'),repeat_annotations=parse('repeat_annotations'))
    result['analysis_source_sha256']=command_manifest()['source_sha256']
    result['groups']=grouped_summary(rows)
    result['automated_evaluable']=automated_evaluable
    result['evaluable_after_review']=sum(r['metrics']['evaluable'] for r in rows)
    result['comparisons']={str(b):rule_results(rows,budget=b,uncertainty=gate['uncertainty_bound'] if gate else 0.) for b in (.01,.03)}
    save_new(a.out,result);print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))
if __name__=='__main__':main()
