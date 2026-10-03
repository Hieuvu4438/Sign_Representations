"""Dataset-specific adapters; source labels are never rewritten."""
import csv
import gzip
import io
import json
import pickle
from pathlib import Path

FIELDS = ['sample_id', 'dataset_id', 'sign_language', 'source_root', 'relative_path',
          'input_kind', 'parent_source_id', 'utterance_id', 'signer_id', 'view_id',
          'official_split', 'research_split', 'start_sec', 'end_sec', 'fps_observed',
          'duration_sec', 'num_frames_observed', 'width', 'height', 'lexical_id',
          'gloss_raw', 'translation_raw', 'metadata_source', 'annotation_quality',
          'decode_status', 'content_hash', 'duplicate_group', 'recording_group']


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Annotation requires executable pickle global {module}.{name}; refused')


def read_labels(path):
    data = Path(path).read_bytes()
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    if data[:1] == b'\x80':
        return PrimitiveUnpickler(io.BytesIO(data)).load(), 'gzip_primitive_pickle'
    try:
        return json.loads(data), 'json_utf8'
    except json.JSONDecodeError:
        return [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()], 'jsonl_utf8'


def sample(dataset, language, root, relative, split, metadata, **values):
    row = dict.fromkeys(FIELDS)
    row.update(dataset_id=dataset, sign_language=language, source_root=str(root),
               relative_path=relative, input_kind='video', official_split=split,
               metadata_source=str(metadata), annotation_quality='GOLD_EXISTING_LEXICAL_OR_SENTENCE',
               decode_status='NOT_CHECKED')
    row.update(values)
    row['sample_id'] = dataset + ':' + Path(relative).stem
    row['recording_group'] = row['recording_group'] or row['sample_id']
    row['source_exists'] = (Path(root) / relative).exists()
    return row


def asl_citizen(root):
    for split in ['train', 'val', 'test']:
        metadata = root / 'splits' / (split + '.csv')
        with metadata.open(encoding='utf-8-sig', newline='') as stream:
            for item in csv.DictReader(stream):
                yield sample('asl_citizen', 'ase', root, 'videos/' + item['Video file'], split, metadata,
                             lexical_id=item['Gloss'], gloss_raw=item['Gloss'], signer_id=item['Participant ID'],
                             lexical_mapping_id=item['ASL-LEX Code'],
                             signer_metadata_source='official CSV Participant ID',
                             utterance_id=Path(item['Video file']).stem)


def wlasl(root):
    metadata = root / 'preprocess/nslt_2000.json'
    records = json.loads(metadata.read_text())
    classes = {}
    for line in (root / 'preprocess/wlasl_class_list.txt').read_text().splitlines():
        index, gloss = line.split(maxsplit=1)
        classes[int(index)] = gloss
    for video_id, item in records.items():
        action = item['action']
        yield sample('wlasl2000', 'ase', root, 'WLASL2000/' + video_id + '.mp4', item['subset'], metadata,
                     lexical_id=classes[action[0]], gloss_raw=classes[action[0]],
                     original_label_id=action[0], source_frame_interval=action[1:],
                     source_frame_interval_status='ANNOTATION_NOT_APPLIED_TO_LOCAL_CLIP',
                     utterance_id=video_id, recording_group='wlasl2000:' + video_id,
                     grouping_status='SOURCE_RECORDING_UNKNOWN')


def msasl(root):
    metadata = root / 'msasl1k.json'
    for item in json.loads(metadata.read_text()):
        video_id = item['video_name']
        yield sample('ms_asl', 'ase', root, 'cut_processed_video/' + video_id + '.mp4', item['subset'], metadata,
                     lexical_id=item['label'], gloss_raw=item['text'], signer_id=str(item['signer_id']),
                     parent_source_id=item['ytb_id'], source_video_url=item['url'],
                     source_interval_sec=[item['start_time'], item['end_time']],
                     source_interval_status='ORIGINAL_VIDEO_TIME_LOCAL_FILE_IS_CROPPED',
                     utterance_id=video_id, recording_group='youtube:' + item['ytb_id'])


def how2sign(root, alternate):
    specifications = [('train', root / 'train/train_label/labels.train'),
                      ('val', root / 'eval/eval_label/labels.dev.json'),
                      ('test', alternate / 'from_uni_sign_source/labels.test')]
    for split, metadata in specifications:
        records, fmt = read_labels(metadata)
        items = records.values() if isinstance(records, dict) else records
        for item in items:
            video = item['video_path']
            parent = item.get('video_id')
            yield sample('how2sign', 'ase', root,
                         {'train': 'train', 'val': 'eval', 'test': 'test'}[split] + '/raw_videos/' + video,
                         split, metadata, translation_raw=item['text'],
                         gloss_raw=item.get('gloss') or None, parent_source_id=parent,
                         utterance_id=item['name'], view_id='rgb_front',
                         source_interval_sec=[item.get('start_time'), item.get('end_time')],
                         source_interval_status='ORIGINAL_RECORDING_TIME_LOCAL_FILE_IS_SENTENCE_CLIP',
                         recording_group=('how2sign:' + parent) if parent else 'how2sign:unresolved:' + video,
                         grouping_status='METADATA_VIDEO_ID' if parent else 'SOURCE_RECORDING_UNKNOWN',
                         annotation_format=fmt)


def csl(root, dataset, video_prefix):
    for split, filename in [('train', 'labels.train'), ('val', 'labels.dev'), ('test', 'labels.test')]:
        metadata = root / filename
        records, fmt = read_labels(metadata)
        for name, item in records.items():
            yield sample(dataset, 'csl_daily_unspecified_variant', root, video_prefix + '/' + item['video_path'],
                         split, metadata, gloss_raw=item['gloss'], translation_raw=item['text'],
                         utterance_id=name, recording_group='csl_daily:' + name,
                         annotation_format=fmt, lineage_status='SHARED_CORPUS_SAMPLE_ID_INTERVAL_UNVERIFIED')


def phoenix(root):
    for split, source_split in [('train', 'train'), ('val', 'dev'), ('test', 'test')]:
        metadata = root / ('phoenix14t.pami0.' + source_split + '.annotations_only.gzip')
        records, fmt = read_labels(metadata)
        for item in records:
            name = item['name']
            video_relative = 'videos_phoenix/videos/' + name + '.mp4'
            frames_relative = 'PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/' + name
            relative = video_relative if (root / video_relative).exists() else frames_relative
            yield sample('phoenix14t', 'gsg', root, relative, split, metadata,
                         input_kind='video' if relative == video_relative else 'frames_directory',
                         signer_id=item['signer'], utterance_id=name,
                         gloss_raw=item['gloss'], translation_raw=item['text'], annotation_format=fmt,
                         recording_group='phoenix14t:' + name.split('/')[-1].rsplit('-', 1)[0],
                         grouping_status='CONSERVATIVE_RECORDING_NAME_CANDIDATE')
