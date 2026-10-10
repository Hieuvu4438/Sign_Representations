from pathlib import Path

from signcoord.data.adapters import how2sign_source,select_candidates


def test_class_selection_is_metadata_only_and_identical_across_train_dev(tmp_path):
    media=tmp_path/'video.mp4';media.write_bytes(b'exists')
    candidates=[{'id':f'{s}:{label}:{i}','split':s,'label':label,'media':media}
                for s in ('train','dev') for label in ('apple','book','car') for i in range(5)]
    train,classes=select_candidates(candidates,'train',max_classes=2,max_per_class=3)
    dev,other=select_candidates(candidates,'dev',max_classes=2,max_per_class=1)
    assert classes==other and len(train)==6 and len(dev)==2
    reversed_train,reversed_classes=select_candidates(list(reversed(candidates)),'train',max_classes=2,max_per_class=3)
    assert [r['id'] for r in train]==[r['id'] for r in reversed_train] and classes==reversed_classes
    assert how2sign_source('youtube_1-8-rgb_front')=='youtube'
