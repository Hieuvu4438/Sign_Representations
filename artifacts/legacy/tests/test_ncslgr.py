import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signrepr.ncslgr import parse_database, inclusive_frame_bounds


XML = '''<SIGNSTREAM-DATABASE>
<PARTICIPANTS><PARTICIPANT ID="0" NAME="Signer"/></PARTICIPANTS>
<CODING-SCHEME>
<FIELD ID="88" NAME="negative"><VALUE ID="2" NAME="neg"/>
<VALUE ID="400001" NAME="ONSET"/></FIELD>
<FIELD ID="16" NAME="POS"><VALUE ID="2" NAME="Negation"/></FIELD>
</CODING-SCHEME>
<MEDIA-FILES><MEDIA-FILE ID="1" LEGACY-PATH="data:video.mov"/></MEDIA-FILES>
<UTTERANCES><UTTERANCE ID="3" S="666" E="3333"><MEDIA-REF ID="1"/>
<SEGMENT PARTICIPANT-ID="0"><TRACK FID="88">
<A S="300" E="2300" VID="2"/><A S="-33" E="266" VID="400001"/>
</TRACK><TRACK FID="16"><A S="0" E="233" VID="2"/></TRACK>
</SEGMENT></UTTERANCE></UTTERANCES></SIGNSTREAM-DATABASE>'''


class NCSLGRTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'sample.xml'
            path.write_text(text)
            return parse_database(path)

    def test_dynamic_field_ids_offsets_and_no_inferred_absence(self):
        data = self.parse(XML)
        u = data['utterances'][0]
        self.assertEqual(u['recorded_functional_labels'], ['negative::neg'])
        self.assertEqual(len(u['functional_core_intervals']), 1)
        event = u['functional_core_intervals'][0]
        self.assertEqual((event['absolute_start_ms'], event['absolute_end_ms']), (966, 2966))
        self.assertIsNone(u['negative_labels'])
        onset = next(e for e in u['events'] if e['phase'] == 'ONSET')
        self.assertEqual(onset['absolute_start_ms'], 633)
        self.assertEqual(data['issues'][0]['kind'], 'event_before_utterance')

    def test_end_frame_is_included_even_for_single_frame_event(self):
        self.assertEqual(inclusive_frame_bounds(1000, 1000, 30), (30, 31))
        self.assertEqual(inclusive_frame_bounds(966, 2966, 30), (29, 90))
        with self.assertRaises(ValueError):
            inclusive_frame_bounds(1000, 900, 30)

    def test_xml_entities_are_not_expanded(self):
        from defusedxml.common import DTDForbidden
        with self.assertRaises(DTDForbidden):
            self.parse('<!DOCTYPE x [<!ENTITY x "external">]>' + XML)

    def test_duplicate_utterance_cannot_silently_overwrite_gold(self):
        event = XML.split('<UTTERANCES>')[1].split('</UTTERANCES>')[0]
        with self.assertRaises(ValueError):
            self.parse(XML.replace('</UTTERANCES>', event + '</UTTERANCES>'))


if __name__ == '__main__':
    unittest.main()
