import unittest
from msasl_metadata_v2 import youtube_id, issues


class ReleaseURLTests(unittest.TestCase):
    def test_explicit_schemeless_extension_and_rejections(self):
        key='Abc_123-xyz'
        for url in ('www.youtube.com/watch?v='+key,'www.youtube.com/watch?feature=share&v='+key,
                    'https://www.youtube.com/watch?v='+key,'https://youtu.be/'+key):
            self.assertEqual(youtube_id(url),key)
        for url in ('www.youtube.com.evil.example/watch?v='+key,'www.youtube.com/watch?v='+key+'&v='+key,
                    'www.youtube.com/watch?v=too-short','youtube.com/watch?v='+key,
                    'www.youtube.com:8443/watch?v='+key,'https://user:pass@youtube.com/watch?v='+key,None):
            self.assertIsNone(youtube_id(url))

    def test_original_interval_guards_preserved(self):
        row=dict(label=0,signer_id=3,signer=-1,url='www.youtube.com/watch?v=Abc_123-xyz',start_time=0.,end_time=2.,
                 fps=30.,start=0,end=60,box=[0.,0.,1.,1.],width=640.,height=480.)
        self.assertEqual(issues(row,1000),[])
        broken=dict(row,end_time=0.,end=0)
        self.assertEqual(set(issues(broken,1000)),{'time_or_FPS_contract','frame_interval_contract'})
        self.assertIn('label_range_or_type',issues(dict(row,label=True),1000))
        self.assertIn('source_url_not_accepted',issues(dict(row,url='www.youtube.com/watch?v=bad'),1000))


if __name__=='__main__':unittest.main()
