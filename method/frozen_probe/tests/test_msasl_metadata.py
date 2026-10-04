import unittest
from msasl_metadata import youtube_id, issues


class MetadataTests(unittest.TestCase):
    def test_source_namespace_parser_and_ambiguous_or_untrusted_urls(self):
        key = 'Abc_123-xyz'
        for url in ('https://www.youtube.com/watch?v='+key, 'http://m.youtube.com/watch?v='+key+'&t=12', 'https://youtu.be/'+key):
            self.assertEqual(youtube_id(url), key)
        for url in ('https://youtube.com.evil.example/watch?v='+key, 'https://youtube.com/watch?v='+key+'&v='+key,
                    'https://user:password@youtube.com/watch?v='+key, 'https://youtu.be/not-an-id',
                    'file:///watch?v='+key, 'https://youtube.com:8443/watch?v='+key, None):
            self.assertIsNone(youtube_id(url))

    def test_annotation_contracts_are_metadata_only(self):
        row = dict(label=0, signer_id=3, signer=-1, url='https://youtu.be/Abc_123-xyz',
                   start_time=0., end_time=2., fps=30., start=0, end=60, box=[0.,0.,1.,1.],width=640.,height=480.)
        self.assertEqual(issues(row, 1000), [])
        for field, value, expected in (('label',True,'label_range_or_type'), ('signer_id',-1,'signer_id_range_or_type'),
                                       ('end_time',0.,'time_or_FPS_contract'), ('fps',float('nan'),'time_or_FPS_contract'),
                                       ('box',[0.,1.,1.,0.],'normalized_box_contract'), ('end',0,'frame_interval_contract')):
            self.assertIn(expected, issues(dict(row, **{field:value}),1000))


if __name__=='__main__':unittest.main()
