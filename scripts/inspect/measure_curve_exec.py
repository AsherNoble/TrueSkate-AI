"""Measure original CURVE-EXEC frames using a command-blind orange extractor."""
import argparse
from pathlib import Path
from trueskate_ai.research.curve_measurement import measure_recording

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();rows=measure_recording(a.recording,a.manifest)
    print(f'{len(rows)} diagnostics; {sum(r["metrics"]["evaluable"] for r in rows)} automatically evaluable')
if __name__=='__main__':main()
