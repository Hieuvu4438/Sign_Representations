from pathlib import Path
import torch
from ..data.pose_dataset import PoseDataset
from ..data.samplers import gloss_key
from ..evaluation.extract import extract_full_sequence
from ..evaluation.retrieval import rank_dictionary
from ..evaluation.representation import representation_diagnostics


@torch.inference_mode()
def frozen_retrieval(encoder, gallery_manifest, query_manifest, config, metric_head=None):
    encoder.eval()
    if metric_head is not None: metric_head.eval()
    datasets = [PoseDataset(path, config['data'], stage='evaluation') for path in (gallery_manifest, query_manifest)]
    matrices = []
    for dataset in datasets:
        embeddings = []
        for index in range(len(dataset)):
            result = extract_full_sequence(encoder, dataset[index], config['evaluation']['window_frames'], config['evaluation']['window_stride_frames'])
            vector = result['raw_embedding']
            if metric_head is not None:
                vector = metric_head(vector[None].to(next(metric_head.parameters()).device))[0].cpu() if result['has_observation'] else torch.zeros(config['model']['latent_dim'])
            embeddings.append(vector)
        matrices.append(torch.stack(embeddings).numpy())
    gallery, query = datasets
    if {r['source_video_id'] for r in gallery.rows} & {r['source_video_id'] for r in query.rows}:
        raise ValueError('Validation retrieval source video contamination')
    result = rank_dictionary(*matrices, [r['sample_id'] for r in gallery.rows], [r['sample_id'] for r in query.rows],
                             [gloss_key(r) for r in gallery.rows], [gloss_key(r) for r in query.rows], config['evaluation']['retrieval_protocol'])
    result['representation_diagnostics'] = representation_diagnostics(matrices[1])
    return result
