# Installation

```
conda create -n skeleton_uplift_3d python=3.10
conda activate skeleton_uplift_3d
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Download Models:
Use this Google Drive link https://drive.google.com/file/d/1mUiO8kNgAYnzWcPKywVz8U89bYy2w2_S/view?usp=sharing to download models and put them in the main directory

```
models/
    ├──img_hand.ckpt
    ├──mlp_body.ckpt
    └──mlp_hand.ckpt
```

# Running Web Camera Demo:
```
python camera_demo.py
```

# Running 3D reconstruction on a single video:
```
python uplift_release.py --video_name <...> --save_file generated/uplift_3d.pt
```

## Show results:
```
python uplift_release.py --save_file generated/uplift_3d.pt --show
```

# LICENSE:

The code subjects to the CC-BY-NC-SA-4.0 License (see LICENSE.md).