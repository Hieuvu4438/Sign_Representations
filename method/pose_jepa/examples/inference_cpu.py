"""CPU inference example for an exported bundle and a standard 133-joint cache."""
import argparse
import json
from signrepr.data.schema import load_pose
from signrepr.export.api import SignRepresentationModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--pose', required=True)
    p.add_argument('--metadata')
    p.add_argument('--embedding', choices=['raw', 'metric'], default='raw')
    args = p.parse_args()
    metadata = json.loads(open(args.metadata).read()) if args.metadata else {'joint_layout': 'coco_wholebody_133', 'coordinate_system': 'viewport_normalized'}
    model = SignRepresentationModel(args.bundle, 'cpu')
    pose = load_pose(args.pose)
    sequence = model.encode_sequence(pose, metadata)
    clip = model.encode_clip(pose, metadata, args.embedding)
    print(json.dumps({'sequence_shape': list(sequence['features'].shape), 'length': sequence['length'],
                      'embedding_shape': list(clip['embedding'].shape), 'status': clip['status'],
                      'selected_encoder': model.metadata['selected_encoder']}))


if __name__ == '__main__': main()
