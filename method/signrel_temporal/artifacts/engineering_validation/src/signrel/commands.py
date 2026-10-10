"""CLI dispatch to implemented pipelines; no fabricated success fallback."""
from .config import load_config


def dispatch(args):
    if not args.config: raise ValueError('--config required')
    if args.command=='train':
        from .train.loop import train
        return train(load_config(args.config).values,args.resume,args.stop_after,args.device,args.dry_run)
    if args.command=='import-data':
        from .data.adapters import import_data
        return import_data(args.config,args.output)
    if args.command=='audit-splits':
        from .data.adapters import audit_data
        return audit_data(args.config,args.output or 'artifacts/split_audit.json')
    if args.command=='extract-cache':
        from .data.extraction import extract_cache
        return extract_cache(load_config(args.config).values,args.limit_videos)
    if args.command=='audit-cache':
        from .pipelines import audit_cache
        return audit_cache(load_config(args.config).values,args.output or 'artifacts/cache_audit.json')
    if args.command=='export-embeddings':
        from .pipelines import export_embeddings
        return export_embeddings(args.config,args.device or 'cpu')
    if args.command=='eval-retrieval':
        from .pipelines import evaluate_retrieval
        return evaluate_retrieval(args.config)
    if args.command=='profile':
        from .pipelines import profile
        return profile(load_config(args.config).values,args.output or 'artifacts/profile.json',args.device)
    if args.command in ('eval-minimal-pairs','aggregate','build-report'):
        from .reporting import minimal_pairs,aggregate,build_report
        return {'eval-minimal-pairs':minimal_pairs,'aggregate':aggregate,'build-report':build_report}[args.command](args.config,args.output)
    raise ValueError('Unknown command')
