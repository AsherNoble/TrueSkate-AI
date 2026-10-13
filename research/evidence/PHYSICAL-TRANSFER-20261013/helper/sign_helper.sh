#!/bin/bash
# Sign the staged unsigned PhysicalTouchLogger with the Personal Team profile that
# Xcode created for the throwaway signing project. Run by the operator in a
# Terminal inside the rig's GUI session (the login keychain must be unlocked).
# Never installs, launches, revokes or rebuilds anything; never touches WebDriverAgent.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
export KIT
exec /usr/bin/python3 -I - "$@" <<'PY'
import datetime as dt, glob, hashlib, json, os, plistlib, re, shutil, subprocess, sys
from pathlib import Path

KIT = Path(os.environ["KIT"])
UNSIGNED = KIT/"unsigned"/"PhysicalTouchLogger.app"
ENV_FILE = Path("/Users/training-server/trueskate-ai/.env")
EXPECTED_UNSIGNED_SHA = "17948b3bc5ba092018cb9bda2a7689d0553945139e1f7fcb98370b3ae732202e"
EXPECTED_TEAM = "BK75JV5YK4"  # the team that signs WDA; another team is reported, not refused
BUNDLES = ("com.trueskate.research.touchlogger.physical20261013",
           "com.ashernoble.touchlogger.physical20261013")
PROFILE_ROOTS = ("~/Library/Developer/Xcode/UserData/Provisioning Profiles",
                 "~/Library/MobileDevice/Provisioning Profiles")


def fail(message):
    print("SIGN_FAIL: "+message, flush=True)
    sys.exit(1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    return subprocess.run(args, capture_output=True, text=True)


if not UNSIGNED.is_dir() or sha256(UNSIGNED/"PhysicalTouchLogger") != EXPECTED_UNSIGNED_SHA:
    fail("staged unsigned helper missing or changed: "+str(UNSIGNED))
if not ENV_FILE.is_file():
    fail("rig environment file missing: "+str(ENV_FILE))
env = {}
for line in ENV_FILE.read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
udid = env.get("IPHONE_XR_UDID", "")
if not udid:
    fail("IPHONE_XR_UDID missing from "+str(ENV_FILE))

now = dt.datetime.now(dt.timezone.utc)
candidates = []
for root in PROFILE_ROOTS:
    for path in glob.glob(os.path.join(os.path.expanduser(root), "*.mobileprovision")):
        decoded = run(["security", "cms", "-D", "-i", path])
        if decoded.returncode:
            continue
        try:
            profile = plistlib.loads(decoded.stdout.encode())
        except Exception:
            continue
        app_id = profile.get("Entitlements", {}).get("application-identifier", "")
        for bundle in BUNDLES:
            if app_id.endswith("."+bundle):
                candidates.append((profile["CreationDate"], path, profile, bundle))
if not candidates:
    fail("no provisioning profile for "+" or ".join(BUNDLES)+"; finish the Xcode build step first")
created, profile_path, profile, bundle = max(candidates, key=lambda c: c[0])
team = profile["TeamIdentifier"][0]
entitlements = profile["Entitlements"]
expires = profile["ExpirationDate"].replace(tzinfo=dt.timezone.utc)
if entitlements["application-identifier"] != team+"."+bundle:
    fail("profile application identifier does not match its team")
if expires-now < dt.timedelta(days=2):
    fail(f"profile expires too soon ({expires.isoformat()})")
if udid not in profile.get("ProvisionedDevices", []):
    fail("profile does not include XR1")
if entitlements.get("get-task-allow") is not True:
    fail("profile is not a development profile (get-task-allow)")

# Sign with a certificate the profile embeds and this keychain can use, so they cannot mismatch.
valid = set(re.findall(r"\b([0-9A-F]{40})\b", run(["security", "find-identity", "-v", "-p", "codesigning"]).stdout))
certificates = {hashlib.sha1(der).hexdigest().upper(): der for der in profile["DeveloperCertificates"]}
usable = sorted(set(certificates) & valid)
if not usable:
    fail("no certificate in the profile has a valid signing identity in this keychain")
identity = usable[0]
subject = subprocess.run(["openssl", "x509", "-inform", "DER", "-noout", "-subject", "-enddate"],
                         input=certificates[identity], capture_output=True).stdout.decode().strip()

out = KIT/("signed-"+now.strftime("%Y%m%dT%H%M%SZ"))
out.mkdir()
app = out/"PhysicalTouchLogger.app"
shutil.copytree(UNSIGNED, app)
info_path = app/"Info.plist"
info = plistlib.loads(info_path.read_bytes())
if info["CFBundleIdentifier"] != bundle:  # Only when Xcode rejected the primary identifier.
    info["CFBundleIdentifier"] = bundle
    info_path.write_bytes(plistlib.dumps(info))
shutil.copy2(profile_path, app/"embedded.mobileprovision")
signed_entitlements = {key: entitlements[key] for key in
                       ("application-identifier", "com.apple.developer.team-identifier", "get-task-allow")}
(out/"entitlements.plist").write_bytes(plistlib.dumps(signed_entitlements))
sign = run(["codesign", "--force", "--sign", identity, "--entitlements", str(out/"entitlements.plist"),
            "--generate-entitlement-der", "--timestamp=none", str(app)])
if sign.returncode:
    hint = " (run this inside the Screen Sharing Terminal so the keychain is unlocked)" \
        if "errSecInternalComponent" in sign.stderr else ""
    fail("codesign failed: "+sign.stderr.strip()+hint)
verify = run(["codesign", "--verify", "--strict", "--verbose=2", str(app)])
if verify.returncode:
    fail("codesign verification failed: "+verify.stderr.strip())
details = run(["codesign", "-d", "--verbose=2", str(app)])
receipt = dict(
    schema="physical-transfer-helper-signing-v1", created_at=now.isoformat(),
    bundle_id=bundle, primary_bundle_id=bundle == BUNDLES[0], team=team,
    team_matches_wda=team == EXPECTED_TEAM,
    profile=dict(name=profile.get("Name"), uuid=profile.get("UUID"), path=profile_path,
                 sha256=sha256(profile_path), created=created.isoformat(),
                 expires=expires.isoformat(), device_count=len(profile.get("ProvisionedDevices", [])),
                 xr1_included=True, xr1_udid_sha256=hashlib.sha256(udid.encode()).hexdigest()),
    certificate=dict(sha1=identity, subject_and_expiry=subject),
    entitlements=signed_entitlements,
    unsigned_executable_sha256=EXPECTED_UNSIGNED_SHA,
    signed_executable_sha256=sha256(app/"PhysicalTouchLogger"),
    info_plist_sha256=sha256(info_path),
    codesign_verify=verify.stderr.strip(), codesign_details=details.stderr.strip(),
    app=str(app), installed=False, launched=False)
(out/"sign-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps({k: receipt[k] for k in ("bundle_id", "team", "team_matches_wda", "app")}
                 | dict(profile_expires=receipt["profile"]["expires"]), indent=2))
print("SIGN_OK "+str(out/"sign-receipt.json"), flush=True)
PY
