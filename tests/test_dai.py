import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signrepr.dai import candidates_from_html, resolve_row


class DAIMappingTests(unittest.TestCase):
    def test_ambiguity_does_not_choose_first_video(self):
        row = {'matched translation': 'A sentence.', 'video_id': 'recording_17'}
        candidate = {'source_translation': 'A sentence.', 'source_recording_name': 'recording',
                     'frontal_video_urls': ['https://example.com/front.mp4'], 'view_label': 'Front',
                     'canonical_utterance_id': 'recording-U-1', 'recording_group': 'asllrp:recording'}
        self.assertEqual(resolve_row(row, [candidate, candidate])['reason'], 'AMBIGUOUS_TRANSLATION_MATCH')

    def test_translation_match_requires_recording_identity(self):
        row = {'matched translation': 'A sentence.', 'video_id': 'wrong_17'}
        candidate = {'source_translation': 'A sentence.', 'source_recording_name': 'recording'}
        self.assertEqual(resolve_row(row, [candidate])['reason'], 'RAW_RECORDING_ID_CONFLICT')

    def test_all_views_map_to_same_recording_and_only_frontal_is_selected(self):
        html = '<td class="datasource" id="ds_1">recording-U-5</td><div id="fg_1">head pos: tilt; engl trans: A sentence.</div><a id="uv_1"></a><div id="uv_1"><source src="/front/42.mp4">U5 View=Front</div><div id="uvf_1"><source src="/face/42.mp4">U5 View=Face</div>'
        candidates, login = candidates_from_html(html, 'https://example.com/page')
        self.assertEqual(candidates[0]['frontal_video_urls'], ['https://example.com/front/42.mp4'])
        self.assertEqual(candidates[0]['recording_group'], 'asllrp:recording')
        result = resolve_row({'matched translation': 'A sentence.', 'video_id': 'recording_17'}, candidates)
        self.assertEqual(result['mapping_status'], 'RESOLVED_METADATA_REVIEW_PENDING')
        self.assertEqual(result['media_download_status'], 'NOT_ATTEMPTED')


if __name__ == '__main__':
    unittest.main()
