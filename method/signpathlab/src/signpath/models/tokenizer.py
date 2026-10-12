from ..contracts import Token, TokenBundle
from ..pose.coordinates import static_pose_streams
from ..pose.topology import POSE_GROUPS


class SignTokenizer:
    def build(self, feature_bundle, geometry_bundle=None) -> TokenBundle:
        b = feature_bundle
        pose = static_pose_streams(b.pose_xy_body, b.pose_quality, b.pose_coordinate_valid)
        tokens = []
        origin = float(b.timestamps[0])
        for t, timestamp in enumerate(b.timestamps):
            for stream in range(7):
                if stream < 4:
                    values = b.visual[t, stream].astype("float32")
                    valid = target_valid = bool(b.visual_valid[t, stream])
                else:
                    joints = POSE_GROUPS[stream - 4]
                    values = pose[stream - 4][t]
                    valid = bool(b.pose_coordinate_valid[t, joints].any())
                    target_valid = bool(b.pose_coordinate_valid[t, joints].all())
                tokens.append(Token(f"{b.meta['sample_id']}:{t}:{stream}", stream, float(timestamp) - origin,
                                    values, valid, target_valid, b.raw_supports[t * 7 + stream]))
        return TokenBundle(b.meta["sample_id"], tuple(tokens), tuple(geometry_bundle or ()))
