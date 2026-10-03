## Data Preparation

### Pretraining Dataset

We pretrain SHuBERT on YouTube-ASL (removing clips which intersect with the test and val sets of OpenASL)
- **Youtube-ASL**: [https://github.com/google-research/google-research/tree/master/youtube_asl](https://github.com/google-research/google-research/tree/master/youtube_asl).
  - **Get the .txt file**: [https://github.com/google-research/google-research/tree/master/youtube_asl](https://github.com/google-research/google-research/tree/master/youtube_asl).
  
  - **Download videos**: 
  `dataset/download_youtubeasl.sh`

  - **Form clips from videos**: 
  `dataset/clips_youtubeasl.sh`

  - **Crop clips**: 
  `dataset/clips_crop.sh`

  - **Get Pose Estimation from Mediapipe**: 
  `dataset/kpe_mediapipe.sh`

  - **Get hands crops**: 
  `dataset/crop_hands.sh` 

  - **Get face crops**: 
  `dataset/crop_face.sh` 

  - **Get body features**: 
  `dataset/body_features.sh`

  - **Get hands features**: 
  `dataset/hands_features.sh`  

  - **Get face features**: 
  `dataset/face_features.sh`


### Downstream Tasks Datasets

- **TODO**

