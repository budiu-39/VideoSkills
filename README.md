# README

## Environment Setup

### 1. Create Conda Environment and Install Dependencies
```bash
git clone git@github.com:budiu-39/GVHMR_PHC.git
cd GVHMR_PHC

conda create -n isaac python=3.8
pip install -r requirement.txt

# Optionally set CUDA path if necessary
# export CUDA_HOME=/usr/local/cuda-11.8/
# export PATH=$PATH:/usr/local/cuda-11.8/bin/
```

### 2. Setup GVHMR Model and Dependencies
```bash
cd GVHMR
pip install -e .

# cd third-party/DPVO
# wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
# unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

mkdir inputs
mkdir outputs
mkdir -p inputs/checkpoints
```

1. You need to register to download [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). Place the models in the following structure:

```text
GVHMR/inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl
```

2. Download other pretrained models from [Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link):
```text
GVHMR/inputs/checkpoints/
├── dpvo/dpvo.pth
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

### 3. Setup PHC Model and Dependencies
You need to register to download [SMPL](https://smpl.is.tue.mpg.de/). Place the models in the following structure:

```text
phc/data
├── smpl
    ├── SMPL_FEMALE.pkl
    ├── SMPL_NEUTRAL.pkl
    ├── SMPL_MALE.pkl
```
### 4. Install Isaac Gym
Download and install [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym).
```bash
cd IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
```

### 5. Download Pretrained G1/H1 Universal Models
```bash
mkdir -p output
gdown https://drive.google.com/uc?id=1MGwaxl3CKnux9UdcwfXBefRh3U6BpIgC -O output/pretrained_model.zip
unzip -o output/pretrained_model.zip -d output/
rm output/pretrained_model.zip
```

## Obtaining Original Motion Data

### 1. From GVHMR (with test video)
```bash
cd GVHMR
python tools/demo/demo_folder.py -f inputs/demo/ -d ../output/GVHMR_output/demo
```

### 2. From Dataset (e.g., AMASS)
Download motion dataset (e.g., AMASS) from official sources or provided links.

## Preprocessing and Retargeting

### Step 1: Fit SMPL Shape
```bash
python scripts/data_process/fit_smpl_shape.py robot=unitree_h1_fitting
python scripts/data_process/fit_smpl_shape.py robot=unitree_g1_fitting
```

### Step 2: Preprocess Motion
- From AMASS:
```bash
python scripts/data_process/fit_smpl_motion.py robot=unitree_h1_fitting +num_jobs=16 +amass_root=../AMASS +robot.process_split=train
python scripts/data_process/fit_smpl_motion.py robot=unitree_g1_fitting +num_jobs=16 +amass_root=../AMASS +robot.process_split=train
```
- From GVHMR output:
```bash
# For SMPL Robot
python scripts/convert_gvhmr_isaac.py --folder_path=GVHMR/outputs/motionx/test_data_136 --output_path=dataset/smpl_motion/136_test
```

Retargeted motions will be stored in `output/Unitree_motion`.

## Train Universal Model

Note: Requires preprocessed AMASS motion data.

```bash
# For SMPL Robot
python videoskills/train.py --task=smpl --use_wandb --headless

# For Unitree G1
python videoskills/train.py --task=g1 --use_wandb --headless
```

Tips:
- Use `--dev` at the end of the command for development/debug mode.
- If you encounter CUDA OOM, reduce `env.num_envs` (e.g., from 2048 to 512).

## Refine Motion Segments
```bash
python videoskills/refine.py --task=smpl --headless --use_wandb --resume
```

```bash
python videoskills/refine.py --task=g1 --headless --use_wandb --resume
```

## Visualize Refined Results

### Render Rollouts (SMPL robots):
```bash
python scripts/render/constrast_render.py --pkl_file /home/miku/Documents/VideoSkills/output/rollout/refinement --use_offscreen
```

### Render Rollouts (Unitree robots):
```bash
python scripts/vis_motion_rollout.py --motion_file=output/HumanoidRef/h1_universal_power0005/rollouts --humanoid_type=h1
```

Results are saved to `output/render_out`.


## `output/` Directory Structure

```text
output/
├── GVHMR_output/               # Output from the GVHMR model (e.g., hmr4d_results.pt, raw motion sequences)
├── HumanoidIm/                 # Output from the Universal Tracker task
│   ├── checkpoint              # Model checkpoints
│   ├── rollout/                # Rollout results
│   └── logs/                   # Training and evaluation logs
├── HumanoidRef/                # Output from the Refine task
│   ├── checkpoint/             # Expert checkpoints for each motion segment
│   ├── rollout/                # Refined motion rollouts
│   └── logs/
├── Humanoid_motion/            # Preprocessed SMPL humanoid motion files
├── Unitree_motion/             # Preprocessed Unitree robot motion files
└── render_out/                 # Rendered rollout videos or images
```

