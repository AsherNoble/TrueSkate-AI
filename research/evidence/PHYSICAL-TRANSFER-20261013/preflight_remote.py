"""Read-only rig preflight. Prints filtered evidence; never opens a device session.

Run over the existing Tailscale SSH connection using the rig's existing .venv.
devicectl's JSON output files are isolated in /private/tmp. Provisioning metadata
is extracted without accepting or installing the profiles.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
from urllib.request import urlopen


def command(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)


def get(path):
    with urlopen("http://127.0.0.1:8100"+path,timeout=5) as r:
        return json.load(r)


base=Path("/Users/training-server/trueskate-ai")
env={}
for line in (base/".env").read_text().splitlines():
    if "=" not in line or line.lstrip().startswith("#"): continue
    k,v=line.split("=",1);env[k.strip()]=v.strip().strip("\"'")
udid=env["IPHONE_XR_UDID"]
temporary=Path("/private/tmp/physical-transfer-preflight-20261013")
temporary.mkdir(exist_ok=True)
appsfile=temporary/"apps.json"
detailsfile=temporary/"device.json"
apps_cmd=command(["xcrun","devicectl","device","info","apps","--device",udid,
                  "--include-all-apps","--json-output",str(appsfile)])
device_cmd=command(["xcrun","devicectl","device","info","details","--device",udid,
                    "--json-output",str(detailsfile)])
apps=json.loads(appsfile.read_text())["result"]["apps"] if not apps_cmd["exit_code"] else []
device=json.loads(detailsfile.read_text())["result"] if not device_cmd["exit_code"] else {}
profiles=[]
for p in sorted(Path("/Users/training-server/Library/Developer/Xcode/UserData/Provisioning Profiles").glob("*.mobileprovision")):
    b=p.read_bytes();start=b.find(b"<?xml");end=b.find(b"</plist>",start)
    d=plistlib.loads(b[start:end+8]);ent=d.get("Entitlements",{})
    profiles.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),name=d.get("Name"),
        created=str(d.get("CreationDate")),expires=str(d.get("ExpirationDate")),
        app_identifier=ent.get("application-identifier"),
        matches_xr1=udid in d.get("ProvisionedDevices",[]),
        metadata_trust="unverified CMS payload metadata only; not a signing authorization"))
disabled=command(["launchctl","print-disabled","gui/501"])
hardware=device.get("hardwareProperties",{})
out=dict(utc=datetime.now(timezone.utc).isoformat(),
    release_symlink=str(base.resolve()),
    release_sha=command(["git","-C",str(base),"rev-parse","HEAD"]),
    release_source_status=command(["git","-C",str(base),"status","--short"]),
    udid_source=str(base/".env")+":IPHONE_XR_UDID",
    xr1_udid_sha256=hashlib.sha256(udid.encode()).hexdigest(),
    details_match_env=hardware.get("udid")==udid,
    device_properties=device.get("deviceProperties",{}),
    hardware={k:hardware.get(k) for k in ("marketingName","productType","hardwareModel","reality")},
    wda_status=get("/status"),foreground=get("/wda/activeAppInfo"),recorder=get("/wda/video"),
    installed_app_count=len(apps),
    relevant_apps=[{k:a.get(k) for k in ("name","bundleIdentifier","version","builtByDeveloper")}
        for a in apps if a.get("builtByDeveloper") or any(s in str(a).lower() for s in ("trueaxis","touchlogger","physicaltouch"))],
    signing_identities=command(["security","find-identity","-v","-p","codesigning"]),
    profiles=profiles,tunnel=command(["launchctl","print","system/com.trueskate.remotexpc-tunnel"]),
    collector_and_watchdog_disabled_lines=[line.strip() for line in disabled["stdout"].splitlines()
        if "com.trueskate.collect" in line or "com.trueskate.watchdog" in line],
    current_park=None,current_waypoint=None,current_settings=None,
    touches_submitted=0,recordings_started=0,helper_installed=False,
    wda_restarted=False,collection_enabled=False)
print(json.dumps(out,indent=2))
