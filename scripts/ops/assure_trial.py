"""Prepare a bounded Assure check; never import or run the control service."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


def write_requirements(output, listing):
    """Export selected canonical API statements, not a claim inferred from our code."""
    profile = next(p for p in listing['profiles'] if p['id'] == 'python-auth-declared-model')
    selected = [s for s in profile['canonical_statements'] if s['id'] in ('AUTH-1', 'AUTH-6')]
    if {s['id'] for s in selected} != {'AUTH-1', 'AUTH-6'}:
        raise ValueError('Assure no longer publishes the selected authentication statements')
    text = '# XR control authentication requirements\n\n'
    text += '\n'.join(f'- R{i}: {s["statement"]}' for i, s in enumerate(selected, 1)) + '\n'
    manifest = {
        'schema': 'trueskate.assure-requirements/v1',
        'source': 'https://api.symbolia.ai/v1/profiles',
        'engine_commit': listing['build']['commit'],
        'profile': profile['id'],
        'profile_sha256': profile['profile_sha256'],
        'requirements_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'status': 'catalogue-derived draft; not inferred from or verified against project source',
        'requirements': [dict(requirement_id=f'R{i}', **s) for i, s in enumerate(selected, 1)],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'requirements.md').write_text(text)
    (output / 'requirements-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, help='Export requirements from an already retrieved API listing')
    parser.add_argument('--output', type=Path, default=Path('tmp/assure'))
    args = parser.parse_args()
    root = Path.cwd()
    output = args.output.resolve()
    if args.profiles:
        write_requirements(output, json.loads(args.profiles.read_text()))
        return 0
    sys.path.insert(0, str(root / 'tmp/assure-action'))
    from action.client import Client, request_oidc_token
    from action import repository_manifest
    from serve.bundle import Collected
    from serve.outcomes import Refusal

    output.mkdir(parents=True, exist_ok=True)

    def write(name, document):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + '\n')

    try:
        token = request_oidc_token(os.environ['ACTIONS_ID_TOKEN_REQUEST_URL'],
                                   os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN'])
        auth = Client('https://api.symbolia.ai', None, sign_in=True).exchange_oidc(token)
        key = auth['key']
        print('::add-mask::' + key)
        with open(os.environ['GITHUB_ENV'], 'a') as env:
            env.write('ASSURE_TRIAL_API_KEY=' + key + '\n')
        client = Client('https://api.symbolia.ai', key)
        listing = client.profiles()
        write('profiles.json', listing)
        write_requirements(output, listing)
        name = 'src/trueskate_ai/control/service.py'
        files = {name: (root / name).read_bytes()}
        for label, declared in [('bundle', 'declaration.json'), ('catalogue', 'catalogue-declaration.json')]:
            declaration = json.loads((root / '.assure' / declared).read_text())
            write(f'{label}/raw/declaration.json', declaration)
            for name, data in files.items():
                path = output / label / 'raw/repository' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            collected = Collected()
            collected.files = {'raw/repository/' + n: data for n, data in files.items()}
            collected.files['raw/declaration.json'] = (output / label / 'raw/declaration.json').read_bytes()
            manifest = repository_manifest.complete(collected, 'python-auth-declared-model',
                                                    output / label, os.environ)
            write(f'{label}/raw/REPOSITORY.json', manifest)
            write(f'{label}-lint.json', client.lint('python-auth-declared-model', collected.files,
                                                  'fails', requirements=(root / 'requirements.md').read_text(),
                                                  compile_requirements=False))
        return 0
    except Refusal as error:
        write('generation-failure.json', error.to_dict())
        print(f'Assure preparation refused: {error.outcome}: {error.reason}')
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write('## Assure preparation did not complete\n\n'
                          f'`{error.outcome}`: {error.reason}\n\n'
                          'Preparation failed. The artifact contains the API-derived requirements draft, '
                          'its local provenance manifest, the API catalogue and the refusal. '
                          'A catalogue statement is not evidence that the service satisfies it.\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
