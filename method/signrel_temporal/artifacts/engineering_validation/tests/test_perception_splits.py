import numpy as np
from signrel.data.perception import body_geometry,associate_hands
from signrel.data.splits import audit_splits,assign_ssl_split


def test_crossing_uses_anatomical_wrists_and_rejects_ambiguity():
    pose=np.full((33,2),.5,np.float32)
    pose[11]=[.3,.3];pose[12]=[.7,.3]
    # Anatomical left wrist is at the right side of this crossed-hands image.
    pose[15]=[.7,.6];pose[16]=[.3,.6]
    right=np.tile(pose[16],(21,1));left=np.tile(pose[15],(21,1))
    assignment,_=associate_hands([right,left],pose,np.ones(33),640,480)
    assert assignment=={1:0,0:1}
    ambiguous=np.full((21,2),[.5,.6])
    assert associate_hands([ambiguous],pose,np.ones(33),640,480)[0]=={}
    assert associate_hands([left,left],pose,np.ones(33),640,480)[0]=={}


def test_body_pixel_aspect_and_invalid_shoulder():
    points=np.full((33,2),.5,np.float32)
    points[11]=[.25,.5];points[12]=[.75,.5];points[0]=[.5,.25]
    body,mask,q,valid,bq,scale=body_geometry(points,np.ones(33),800,400)
    assert valid and scale==400 and body[1]==-.25
    assert mask.all()
    points[12]=points[11]
    body,mask,q,valid,bq,scale=body_geometry(points,np.ones(33),800,400)
    assert not valid and not mask.any() and not body.any()


def test_split_audit_rejects_eval_content_duplicates():
    rows=[{'video_id':'g','sha256':'same','source_recording_id':'g','signer_id':'a','eval_role':'gallery','official_split':'train'},
          {'video_id':'q','sha256':'same','source_recording_id':'q','signer_id':'b','eval_role':'query','official_split':'val'}]
    assert audit_splits([],rows)['status']=='failed'
    assert assign_ssl_split('one',2026)==assign_ssl_split('one',2026)
