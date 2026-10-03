"""Reconnect local upstream sources and repair relocated virtual environments."""
import json
from pathlib import Path


def main():
    legacy = Path(__file__).resolve().parents[1]
    repository = legacy.parents[1]
    source = repository / 'third_party'
    if not source.is_dir():
        raise FileNotFoundError(f'Clone the registered upstream sources into {source}')
    link = legacy / 'third_party'
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ValueError(f'Unexpected source link: {link}')
    elif link.exists():
        raise FileExistsError(f'Refusing to replace {link}')
    else:
        link.symlink_to('../../third_party', target_is_directory=True)

    repaired = []
    environments = []
    for environment in sorted(legacy.glob('.venv-*')):
        configuration = environment / 'pyvenv.cfg'
        if not configuration.is_file():
            continue
        environments.append(environment.name)
        # Activation scripts, executable shebangs and editable dependency paths
        # retain the pre-move location. Leave upstream paths and packages intact.
        files = list((environment / 'bin').glob('*'))
        for packages in environment.glob('lib/python*/site-packages'):
            for pattern in ('*.pth', '*.egg-link', '__editable__*finder.py'):
                files.extend(packages.glob(pattern))
        replacements = {
            str(repository / name): str(legacy / name)
            for name in [p.name for p in legacy.glob('.venv-*')] + ['src']
        }
        for path in files:
            if path.is_symlink() or not path.is_file():
                continue
            try:
                original = path.read_text()
            except UnicodeError:
                continue
            updated = original
            for old, new in replacements.items():
                updated = updated.replace(old, new)
            if updated != original:
                path.write_text(updated)
                repaired.append(str(path.relative_to(legacy)))
        if not (environment / 'bin/python').exists():
            raise FileNotFoundError(f'Base Python missing for {environment}; recreate this environment')
    print(json.dumps({'source_link': str(link), 'environments': environments,
                      'repaired_files': repaired}, indent=2))


if __name__ == '__main__':
    main()
