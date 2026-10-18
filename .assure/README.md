# Assure trial

This optional GitHub Actions trial uploads only the public source at
`src/trueskate_ai/control/service.py`, its explicit model declarations and
`requirements.md`. It never imports or executes the service and needs no device,
database, private environment file or rig connection.

The preparation step signs in with GitHub OIDC and exports the API's canonical
requirements catalogue. The pinned Action then makes two separate API checks:

1. **Controller adapter:** `declaration.json` selects `python/stdlib-http`, the
   actual controller source root, its four public static assets and the launch
   token's API-key credential category. Registration completeness is false.
   The custom header comparison and handler dispatch have not been modelled.
2. **Unmodelled catalogue diagnostics:** `catalogue-declaration.json` selects the
   documented legacy `flask-session-decorator` reader but declares no app objects
   or guards. The controller does not use Flask; this second check requests
   missing-model diagnostics only. It provides no Flask or controller security
   coverage and must never be interpreted as establishing either requirement.

Each check has its own verdict file. The `assure-control-auth` workflow artifact
retains these, both lint records, the API catalogue, source manifests and fresh
requirements draft for seven days. Read the first line, coverage and gaps. This
workflow is not a required merge check; existing controller tests remain
authoritative. A failed or yellow check is deliberately kept visible.

## Beta limitations found

The [initial hosted trial](https://github.com/AsherNoble/TrueSkate-AI/actions/runs/39094775879)
authenticated but the API refused `python/stdlib-http` declaration generation:
its proposer supports only Flask, FastAPI and Django. The repository bot-PR
setting was enabled and verified; that cannot add a server-side proposer.

An explicit standard-library contract bypasses that missing proposer. The
[schema-valid submission](https://github.com/AsherNoble/TrueSkate-AI/actions/runs/39095744719)
then returned `profile_refused`, `definition_absent (route_table_module)`:
the reader expected a parsed route-table module. This service routes inside
handler methods. No application code was changed to accommodate the checker.
The second, deliberately unmodelled check is kept separate from this refusal.

The [gap-reporting run](https://github.com/AsherNoble/TrueSkate-AI/actions/runs/39095949448)
returned a real `assure.serve.verdict/v1` record for that second check:
yellow, `no_verdict`, zero of one machines read, all six obligations not
established, and R1/R2 both `not_established` with `missing_premise`. Its
check id is `28c7d07bf460589e6ee25e358df766cc`. Nothing was established. Missing
`app_objects` and `guards` are named explicitly; the session-related obligations
also name their missing identity/session declarations. The diagnostic step exits
3 rather than turning an unread model green.

The identity contract's schema is referenced through an unshipped API quickstart.
The structural fields were checked against API validation, using a clearly
labelled synthetic FastAPI proposal only to inspect the shared contract shape.
That synthetic proposal is not part of either controller check.

## Requirements and manifests

The authenticated API route catalogue exposes no requirements-generation
endpoint. Assure's public `assure-requirements-typed.json` generator is specific
to PostgreSQL; it is not a Python authentication manifest.

`requirements.md` is generated from canonical AUTH-1 and AUTH-6 statements.
`requirements-manifest.json` records their API provenance and selection; it is a
local format, not Assure's PostgreSQL typed companion. These are reviewable goals,
not observations. Lint matched R1 to AUTH-1 and R2 to AUTH-6. Other catalogue
statements concerning account roles, second factors, sign-out and elevation were
omitted because the controller uses a launch token rather than user accounts.

Regenerate the draft and manifest from a downloaded `profiles.json` artifact:

```sh
python scripts/ops/assure_trial.py --profiles /absolute/path/profiles.json --output tmp/assure-draft
```

Review before replacing `requirements.md` and
`.assure/requirements-manifest.json`. Each run exports a fresh draft without
overwriting committed intent. The two `raw/REPOSITORY.json` manifests are computed
over exactly the source bytes uploaded, including the standard-library
declaration digest where required.

Reports, model-based requirements compilation and paid overage are disabled.
See [Assure's data policy](https://github.com/Symbolia-Assurance/assure-action/blob/69f3d485f01cb41cace7b9cd57f4503c378fa1b3/docs/DATA.md).

Recommendation: use these outputs to identify coverage gaps. Before adopting
Assure as a controller check, obtain an adapter/model for the actual custom
header guard and method-based dispatcher, then review the generated assumptions.

Removal: delete `.github/workflows/assure.yml`, `scripts/ops/assure_trial.py`,
`.assure/` and `requirements.md`. No stored API key or recurring schedule was
added. The personal skill is separate. Local removal does not erase GitHub
artifacts or Symbolia's retained records.
