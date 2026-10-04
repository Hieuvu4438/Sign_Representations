"""Freeze target support/query/extraction IDs once, before all model quality."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from dictionary_sampling import sample_galleries


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;root=Path(c['cohort_runtime']);summary=json.loads((root/'adapted_cohort_summary.json').read_text())
        assert summary['no_incomplete_rows'] and summary['all100classes_eligible'] and summary['accounted_rows']==3425
        rows=json.loads((root/'adapted_cohort_records.json').read_text());eligible=[x for x in rows if x['status']=='eligible']
        frozen=sample_galleries(eligible,list(range(100)),tuple(c['gallery_seeds']))
        byid={x['sample_id']:x for x in rows};candidate=json.loads(Path(c['candidate_records_path']).read_text())['candidates'];candidate={x['sample_id']:x for x in candidate}
        extraction_ids=sorted(set(frozen['query_ids'])|{id for ids in frozen['supports'].values() for id in ids})
        manifest=[]
        for id in extraction_ids:
            x=byid[id];meta=candidate[id];n=x['native']
            manifest.append(dict(sample_id=id,split=x['split'],class_id=x['class_id'],signer_id=x['signer_id'],parent_id=x['parent_id'],
                annotation_index=x['annotation_index'],raw_path=meta['raw_path'],source_sha256=x['source_sha256'],
                start_time_seconds=meta['start_time_seconds'],end_time_seconds=meta['end_time_seconds'],box_yxyx=meta['box_yxyx'],
                time_base=n['time_base'],origin_pts=n['origin_pts'],selected_native_frames=n['selected_native_frames'],
                complete_stride2_windows=n['complete_stride2_windows'],clock_pts_ticks_sha256=n['clock_pts_ticks_sha256'],
                all_resized_RGB_sha256=n['all_resized_RGB_sha256'],first_native16_RGB_sha256=n['first_native16_RGB_sha256'],
                crop_yxyx_pixels=n['crop_yxyx_pixels'],input_contract=x['input_contract'],validation_origin=x['validation_origin']))
        queries=[byid[x] for x in frozen['query_ids']];groups=defaultdict(set)
        for x in queries:groups[x['parent_id']].add(x['signer_id'])
        assert all(len(v)==1 for v in groups.values())
        supports={s:dict(rows=len(ids),unique_parents=len({byid[id]['parent_id'] for id in ids}),
                        signers=len({byid[id]['signer_id'] for id in ids}),
                        affected_estimated_clock_rows=sum(byid[id]['prior_raw_clock_status']=='incomplete' for id in ids)) for s,ids in frozen['supports'].items()}
        out=dict(extraction_rows=len(manifest),unique_support_rows=len(extraction_ids)-len(queries),query_rows=len(queries),
            query_signers=len({x['signer_id'] for x in queries}),query_parents=len(groups),query_class_counts=dict(Counter(x['class_id'] for x in queries)),
            supports=supports,selected_native_frames=sum(x['selected_native_frames'] for x in manifest),
            complete_stride2_windows=sum(x['complete_stride2_windows'] for x in manifest),
            max_native_frames=max(x['selected_native_frames'] for x in manifest),
            affected_extraction_rows=sum(byid[id]['prior_raw_clock_status']=='incomplete' for id in extraction_ids),
            label_budget=dict(target_gallery_labels_per_seed=100,seeds=c['gallery_seeds'],extra_ASLCitizen_geometry_fit_labeled_clips=1313,extra_base_classes=100,
                             target_fit='none; labels used only in fixed dictionary/evaluator'),
            input_contract='MSASL_ESTCLK_v1',pretraining_overlap='unknown; publicsourcepool absence is not checkpoint training exclusion',
            GPU_hours=0,new_encoder_calls=0,quality_access='none',quality_gate='unregistered; no taskperformance',
            interpretation='Fixed supports shared by all methods, no reselection/targetfit. Original3short excluded beforequality, all748eligibleval retained. Adapted dictionary not officialclassifier benchmark.')
        run.save('dictionary_manifest.json',frozen);run.save('extraction_manifest.json',manifest);run.save('dictionary_summary.json',out)


if __name__=='__main__':main()
