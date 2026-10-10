import numpy as np
from PIL import Image

from signcoord.data.perceptual import candidate_matches,phash


def test_perceptual_reencoding_resize_and_temporal_matching():
    rng=np.random.default_rng(7)
    image=rng.integers(0,256,size=(120,160,3),dtype=np.uint8)
    resized=np.asarray(Image.fromarray(image).resize((320,240),Image.Resampling.BILINEAR))
    assert (phash(image)^phash(resized)).bit_count()<=6
    frames=[{'hashes':[1,2,3,4,5]}]
    assert candidate_matches(frames,[{'hashes':[1,2,3,4,5]}],max_distance=0)[0]['minimum_hamming']==0
    assert not candidate_matches(frames,[{'hashes':[1,255,255,255,255]}],max_distance=0)
    assert not candidate_matches(frames,[{'hashes':[5,4,3,2,1]}],max_distance=0)
    # Search indexing must retain every pair at <=6 bits (7 disjoint blocks).
    base=rng.integers(0,2**63,size=5,dtype=np.int64).tolist()
    near=[x ^ sum(1<<b for b in (0,11,23,32,45,59)) for x in base]
    assert candidate_matches([{'hashes':base}],[{'hashes':near}])[0]['minimum_hamming']==6
