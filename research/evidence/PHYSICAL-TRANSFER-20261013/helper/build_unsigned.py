"""Build an isolated unsigned arm64 app. Does not sign, install or launch anything."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
app=args.out/"PhysicalTouchLogger.app"
app.mkdir(parents=True,exist_ok=False)
sdk=subprocess.check_output(["xcrun","--sdk","iphoneos","--show-sdk-path"],text=True).strip()
source_sha=hashlib.sha256((root/"main.m").read_bytes()).hexdigest()
cmd=["xcrun","--sdk","iphoneos","clang","-arch","arm64","-isysroot",sdk,
     "-miphoneos-version-min=18.0","-fobjc-arc","-fblocks","-O2","-Wall","-Wextra",
     '-DLOGGER_SOURCE_SHA256="'+source_sha+'"',
     "-Wno-unused-parameter", "-framework","UIKit","-framework","QuartzCore",
     str(root/"main.m"),"-o",str(app/"PhysicalTouchLogger")]
r=subprocess.run(cmd,capture_output=True,text=True)
shutil.copy2(root/"Info.plist",app/"Info.plist")
record=dict(command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,
            sdk=sdk,signed=False,installed=False,launched=False)
if not r.returncode:
 record["binary_sha256"]=hashlib.sha256((app/"PhysicalTouchLogger").read_bytes()).hexdigest()
record["source_sha256"]=source_sha
(args.out/"build.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
raise SystemExit(r.returncode)
