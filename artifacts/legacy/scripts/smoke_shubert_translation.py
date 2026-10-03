"""Local CPU author-demo11625 numerical smoke on one historical source-captioned clip."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from _common import ROOT, relocated_local_path
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.scoring import token_nll
from signrepr.shubert import load_native_dino
from signrepr.timestamps import source_frame_intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve translation smoke attempts')
    args.output.mkdir(parents=True)
    begun = time.perf_counter()
    write_json(args.output / 'status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'device': 'cpu'})
    try:
        torch.set_num_threads(2)
        registry = json.loads((ROOT / 'provenance/assets.json').read_text())['assets']
        verified = {}
        needed = ['W08', 'W03', 'W04', 'W05', 'W06'] + [k for k in registry if k.startswith('C04_') or k.startswith('W08_CONFIG_')]
        for name in needed:
            value = registry[name]
            path = ROOT / value['local_path']
            if sha256(path) != value['expected_sha256']:
                raise ValueError('Unverified author-demo dependency: ' + name)
            verified[name] = path
        source = ROOT / 'third_party/shubert-author-demo'
        sys.path.insert(0, str(ROOT / 'third_party/SHuBERT/fairseq'))
        sys.path.insert(0, str(source))
        import fairseq
        import inference
        from transformers import ByT5Tokenizer
        # No app/UI or processor download wrappers are imported.
        modules = {}
        for name in ['kpe_mediapipe', 'crop_face', 'crop_hands', 'body_features']:
            spec = importlib.util.spec_from_file_location('demo_' + name, source / (name + '.py'))
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
            modules[name] = loaded
        manifest = ROOT / 'data/external/ncslgr/diagnostic_v1/manifest.jsonl'
        locked = json.loads((manifest.parent / 'protocol.lock.json').read_text())
        if sha256(manifest) != locked['manifest_sha256']:
            raise ValueError('Locked captioned diagnostic manifest changed')
        row = next(r for _, r in read_jsonl(manifest)
                   if r['utterance_sample_id'] == 'ncslgr:ncslgr10a:28' and r['view_role'] == 'body')
        annotation = next(r for _, r in read_jsonl(ROOT / 'reports/ncslgr_annotation_audit/utterances.jsonl')
                          if r['sample_id'] == row['utterance_sample_id'])
        captions = [e['text'] for e in annotation['events'] if e['field_name'] == 'English translation' and e['text']]
        if len(captions) != 1:
            raise ValueError('Smoke requires one unambiguous source-recorded caption')
        path = relocated_local_path(Path(row['source_root']) / row['relative_path'])
        sidecar = json.loads(path.with_suffix('.metadata.json').read_text())
        if sha256(path) != sidecar['prepared_sha256'] or not sidecar['roi_lossless_verified']:
            raise ValueError('Prepared source-captioned clip failed ROI/checksum provenance')
        # Prepared input integrity comes from its recorded lossless ROI conversion.
        import decord
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        indices = list(range(len(reader)))
        if not 1 <= len(indices) <= 200:
            raise ValueError('CPU numerical smoke is bounded to200frames')
        frames = reader.get_batch(indices).asnumpy()
        clock, tail_policy = source_frame_intervals(path, len(frames), reader.get_avg_fps())
        landmarks = modules['kpe_mediapipe'].video_holistic(
            frames, str(verified['W05']), str(verified['W06']))
        hand_extractor = modules['crop_hands'].HandExtractor()
        left, right = hand_extractor.extract_hand_frames(frames, landmarks)
        faces = modules['crop_face'].FaceExtractor().extract_face_frames(frames, landmarks)
        pose = modules['body_features'].process_pose_landmarks(landmarks)
        config = yaml.safe_load((ROOT / 'configs/shubert.yaml').read_text())
        face_model, face_audit = load_native_dino(ROOT, config, 'W03', device='cpu')
        hand_model, hand_audit = load_native_dino(ROOT, config, 'W04', device='cpu')
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        def embed(model, values):
            features = []
            with torch.inference_mode():
                for start in range(0,len(values),16):
                    x = torch.stack([transform(Image.fromarray(frame))[:3] for frame in values[start:start+16]])
                    features.append(model(x).numpy())
            return np.concatenate(features)
        streams = {'face_features': embed(face_model, faces), 'left_hand_features': embed(hand_model,left),
                   'right_hand_features': embed(hand_model,right), 'pose_features': pose.astype(np.float32)}
        if any(value.shape != (len(frames),14 if name=='pose_features' else 384) or not np.isfinite(value).all()
               for name,value in streams.items()):
            raise ValueError('Demo cue stream shapes/values differ')
        np.savez_compressed(args.output / 'demo_streams.npz', **streams,
                            sampled_indices=np.asarray(indices), source_frame_intervals_sec=clock)
        print(json.dumps({'stage':'DEMO_REAL_VIDEO_STREAMS_PASS','frames':len(frames)}),flush=True)
        del face_model, hand_model
        weights = torch.load(verified['W08'], map_location='cpu', weights_only=True)
        model_config = inference.SignLanguageByT5Config.from_pretrained(str(verified['W08'].parent), local_files_only=True)
        model = inference.SignLanguageByT5ForConditionalGeneration(model_config)
        load = model.load_state_dict(weights, strict=True)
        encoder_prefix = 'encoder.adapter.signhubert_adapter.signhubert.'
        embedded = {key[len(encoder_prefix):]: value for key,value in weights.items() if key.startswith(encoder_prefix)}
        pretrained = torch.load(ROOT / config['checkpoint_path'], map_location='cpu', weights_only=False)['model']
        common = sorted(set(embedded) & set(pretrained))
        differences = [key for key in common if embedded[key].shape != pretrained[key].shape
                       or not torch.equal(embedded[key], pretrained[key])]
        encoder_compare = {'demo_encoder_keys':len(embedded),'native_encoder_keys':len(pretrained),
                           'common_keys':len(common),'different_keys':len(differences),
                           'first_different_keys':differences[:10],
                           'only_demo_keys':sorted(set(embedded)-set(pretrained)),
                           'only_native_keys':sorted(set(pretrained)-set(embedded))}
        del weights, embedded, pretrained
        model.eval().requires_grad_(False)
        tokenizer = ByT5Tokenizer.from_pretrained(str(ROOT / 'checkpoints/shubert_author_hf/models/byt5_base'), local_files_only=True)
        tokens = tokenizer(captions[0], return_tensors='pt')['input_ids']
        if tokens[0,-1].item() != tokenizer.eos_token_id:
            raise ValueError('Source caption EOS token policy changed')
        inputs = {key:torch.from_numpy(value).unsqueeze(0) for key,value in streams.items()}
        inputs['attention_mask'] = torch.ones((1,len(frames)),dtype=torch.long)
        with torch.inference_mode():
            first = model(**inputs,labels=tokens,return_dict=True)
            measured = token_nll(first.logits,tokens)
            padded = torch.nn.functional.pad(tokens,(0,8),value=-100)
            repeated = model(**inputs,labels=padded,return_dict=True)
            padded_score = token_nll(repeated.logits,padded)
            difference = float(torch.abs(first.logits-repeated.logits[:,:tokens.shape[1]]).max())
            if difference>1e-5 or abs(float(first.loss)-float(measured['mean_nll'][0]))>1e-5:
                raise ValueError('Teacher-forced padding/recomputation agreement failed')
            if abs(float(measured['sum_nll'][0])-float(padded_score['sum_nll'][0]))>1e-4:
                raise ValueError('Padding changed valid-token NLL')
            generated = model.generate(**inputs,max_length=49,num_beams=1,do_sample=False)
        report = {'status':'PASS_LOCAL_PUBLIC_DEMO_NUMERICAL_SMOKE','model_name':'SHuBERT-author-demo-11625',
                  'device':'cpu','elapsed_seconds':time.perf_counter()-begun,'sample_id':row['utterance_sample_id'],
                  'video_sha256':sha256(path),'source_annotation_sha256':annotation['annotation_sha256'],
                  'sampled_frame_indices':indices,'sampled_frames':len(frames),'source_clock_tail_policy':tail_policy,
                  'source_caption':captions[0],'caption_policy':'Source-recorded English translation; one numerical smoke, no semantic minimal-pair accuracy.',
                  'sum_nll':float(measured['sum_nll'][0]),'token_count':int(measured['token_count'][0]),
                  'mean_nll':float(measured['mean_nll'][0]),'native_mean_loss':float(first.loss),
                  'padding_agreement_max_logit_difference':difference,'eos_scored':True,'padding_label':-100,
                  'generation_text':tokenizer.decode(generated[0],skip_special_tokens=True),
                  'generation_token_ids':generated[0].tolist(),'generation_limit_total_tokens':49,
                  'strict_load':True,'missing_keys':list(load.missing_keys),'unexpected_keys':list(load.unexpected_keys),
                  'encoder_W02_comparison':encoder_compare,'dino_load_audits':[face_audit,hand_audit],
                  'trainable_parameters':0,'total_parameters':sum(p.numel() for p in model.parameters()),
                  'checkpoint_sha256':registry['W08']['expected_sha256'],
                  'implementation_sha256':sha256(Path(__file__)),
                  'source_revision':'69d3d77aa4a4fec89048f5417d44fa717e36f6f8',
                  'source_file_sha256':{key:sha256(path) for key,path in verified.items() if key.startswith('C04_')},
                  'runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__},
                  'limits':['One training-signer source-captioned clip; no grammar accuracy, matched/mismatched semantic pair evaluation or ASL-MTP reproduction.',
                            'Prepared frontal ROI is footer-cropped; C04 native cue extraction/stride1 used locally. No external upload.',
                            'Runtime torch2.1.1 differs from author2.2; observed compatibility only, not bitwise reproduction.',
                            'Author missing-landmark carry-forward/black crops retained. Decoder pipeline result does not establish frozenW02 information loss.']}
        write_json(args.output / 'report.json',report)
        write_json(args.output / 'status.json',{'status':report['status'],'elapsed_seconds':report['elapsed_seconds']})
        print(json.dumps({k:report[k] for k in ['status','mean_nll','token_count','generation_text','elapsed_seconds']}))
    except Exception as error:
        write_json(args.output / 'status.json',{'status':'FAIL','failure_reason':f'{type(error).__name__}: {error}',
                                              'elapsed_seconds':time.perf_counter()-begun})
        raise


if __name__=='__main__':
    main()
