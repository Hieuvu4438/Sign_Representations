from ..cache.hashing import canonical_hash

UPPER7_MEDIAPIPE = (0, 11, 12, 13, 14, 15, 16)
HAND_NAMES = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
              "index_mcp", "index_pip", "index_dip", "index_tip",
              "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
              "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
              "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip")
JOINT_NAMES = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_body_wrist", "right_body_wrist") + tuple("left_" + n for n in HAND_NAMES) + tuple("right_" + n for n in HAND_NAMES)
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
EDGES = ((1, 2), (1, 3), (3, 5), (2, 4), (4, 6)) + tuple((a + offset, b + offset) for offset in (7, 28) for a, b in HAND_EDGES)
POSE_GROUPS = (tuple(range(7)), tuple(range(7, 28)), tuple(range(28, 49)))
TOPOLOGY_HASH = canonical_hash({"version": "upper7_hands42_v1", "names": JOINT_NAMES, "edges": EDGES, "mediapipe_upper": UPPER7_MEDIAPIPE})
