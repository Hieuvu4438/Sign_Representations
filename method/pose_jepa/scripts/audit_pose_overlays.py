"""Produce auditable native-frame pose overlays and joint quality diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from signrepr.data.schema import read_manifest, load_pose, PART_INDICES


def main():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=20)
    args = p.parse_args()
    rows = read_manifest(args.manifest)[:args.samples]
    args.output.mkdir(parents=True, exist_ok=True)
    colors = ['cyan', 'lime', 'yellow', 'magenta']
    audit = []
    for page_start in range(0, len(rows), 5):
        page = rows[page_start:page_start+5]
        fig, axes = plt.subplots(len(page), 4, figsize=(12, len(page)*2.4), squeeze=False)
        for i, row in enumerate(page):
            pose = load_pose(row['pose_path'])
            distance = np.linalg.norm(pose['xy'][:, 91]-pose['xy'][:, 112], axis=-1)
            selected = sorted(set([0, len(distance)//3, int(distance.argmin()), len(distance)-1]))
            while len(selected) < 4: selected.append(selected[-1])
            video = cv2.VideoCapture(row['video_path'])
            for j, frame_index in enumerate(selected[:4]):
                video.set(cv2.CAP_PROP_POS_FRAMES, int(pose['frame_indices'][frame_index]))
                ok, frame = video.read()
                if not ok: raise ValueError(f'Cannot inspect source frame: {row["sample_id"]}')
                h, w = frame.shape[:2]
                axes[i, j].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for color, ids in zip(colors, PART_INDICES.values()):
                    valid = pose['observed'][frame_index, ids] & (pose['confidence'][frame_index, ids] >= .3)
                    xy = pose['xy'][frame_index, ids][valid]*[w, h]
                    axes[i, j].scatter(xy[:, 0], xy[:, 1], s=4, c=color, alpha=.65)
                axes[i, j].set_title(f"{row['gloss_id']} / {row['signer_id']} / frame {frame_index}", fontsize=8)
                axes[i, j].axis('off')
            video.release()
            audit.append({'sample_id': row['sample_id'], 'frames_inspected': selected[:4],
                          'minimum_hand_root_distance_viewport': float(distance.min()),
                          'subthreshold_joint_fraction': float((pose['confidence'] < .3).mean()),
                          'multiple_people_audit': 'see conversion tracking report',
                          'overlay_note': 'verify identity/occlusion visually; confidence is uncalibrated and does not prove visibility'})
        fig.tight_layout()
        fig.savefig(args.output/f'overlays_{page_start//5+1}.jpg', dpi=110)
        plt.close(fig)
    (args.output/'audit.json').write_text(json.dumps({'samples': len(rows), 'rows': audit, 'legend': dict(zip(PART_INDICES, colors))}, indent=2))
    print(json.dumps({'samples': len(rows), 'overlay_pages': (len(rows)+4)//5}))


if __name__ == '__main__': main()
