"""Uniform CLI; structured errors never masquerade as successful phases."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m signrel')
    commands = p.add_subparsers(dest='command',required=True)
    names = ['doctor','resolve-sources','import-data','audit-splits','extract-cache','audit-cache',
             'make-synthetic','profile','train','export-embeddings','eval-retrieval','eval-minimal-pairs','aggregate','build-report']
    for name in names:
        sub = commands.add_parser(name)
        sub.add_argument('--config')
        sub.add_argument('--output')
        sub.add_argument('--dry-run',action='store_true')
        sub.add_argument('--limit-videos',type=int)
        sub.add_argument('--resume')
        sub.add_argument('--device',choices=['auto','cpu','cuda'])
        if name=='make-synthetic': sub.add_argument('--seed',type=int,default=17)
        if name=='train': sub.add_argument('--stop-after',type=int)
    args = p.parse_args(argv)
    try:
        if args.dry_run and args.command!='train':
            import yaml
            config=yaml.safe_load(Path(args.config).read_text()) if args.config else None
            print(json.dumps({'status':'dry_run','command':args.command,'configuration_parsed':config is not None,
                              'execution_started':False,'artifacts_written':False}))
            return 0
        if args.command=='doctor':
            from .doctor import doctor
            result = doctor(args.output or 'artifacts/environment.json')
        elif args.command=='make-synthetic':
            from .data.synthetic import make_synthetic
            result = make_synthetic(args.output or 'data/synthetic',args.seed)
        elif args.command=='resolve-sources':
            from .sources import resolve_sources
            result = resolve_sources(args.config or 'configs/sources.yaml',args.output or 'artifacts/sources.lock.json',args.device or 'cpu')
        else:
            from .commands import dispatch
            result = dispatch(args)
        print(json.dumps(result,allow_nan=False))
        return 0
    except Exception as exc:
        error = {'status':'failed','command':args.command,'error_type':type(exc).__name__,'error':str(exc)}
        print(json.dumps(error),file=sys.stderr)
        from .utils import atomic_json
        atomic_json('artifacts/errors/'+args.command+'.json',error)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
