from signcoord.data.manifest import audit_overlap


def record(sid,source,media,signer=None,start=0,end=1000):
    return {'sample_id':sid,'dataset_id':'fixture','source_video_id':source,'source_group_id':source,
            'media_sha256':media,'signer_id':signer,'start_ms':start,'end_ms':end}


def test_overlap_media_intervals_and_recording_groups():
    a=record('a','recordingA','hash1','P1')
    b=record('b','recordingB','hash1','P2')
    result=audit_overlap([a],[b])
    assert not result['passed'] and result['exact_source_session_signer_collisions'][0]['media_interval']
    b=record('b','recordingA','hash2','P2',start=1100,end=2100)
    result=audit_overlap([a],[b])
    assert not result['passed'] and result['exact_source_session_signer_collisions'][0]['session']


def test_unknown_signers_never_verify_signer_disjoint():
    result=audit_overlap([record('a','A','1')],[record('b','B','2')])
    assert result['passed'] and not result['signer_disjoint_verified']
    assert result['near_duplicate_verification']=='not_run'
