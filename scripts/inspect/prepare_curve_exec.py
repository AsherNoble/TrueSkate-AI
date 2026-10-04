"""Freeze CURVE-EXEC protocol/commands and produce device-free approximation evidence."""
import argparse
from pathlib import Path
from trueskate_ai.research.curve_protocol import command_manifest,offline_report,save_new

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('use a new output directory')
    save_new(a.out/'manifest.json',command_manifest())
    save_new(a.out/'offline.json',offline_report())
    print(a.out)
if __name__=='__main__':main()
