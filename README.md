# VideoSkills

An end-to-end pipeline: feed in a video of human motion, get out a controller that makes a robot perform the same motion --- across multiple robot platforms.

[中文版 README](README.zh-CN.md)

## Features

| Feature                                              | Status |
|------------------------------------------------------|--------|
| Training a universal motion-tracking model           | ✅     |
| Training expert models for single/multiple motions   | ✅     |
| Motion retargeting                                   | ✅     |
| Motion denoising / refinement                        | ✅     |
| Language-conditioned motion control                  | 🔜 Coming soon |
| Deployment on real robots                            | 🔜 Coming soon |

![Demo](demo/dance.gif)
Unitree G1 learns to dance from a video.
![Demo](demo/gym.gif)
Motion denoising: raw motion-estimator output (red character) vs. the refined result (green character).



## Supported robot platforms

| Platform    | Status     |
|-------------|------------|
| SMPL robot  | ✅          |
| Unitree G1  | ✅          |
| Unitree H1  | 🔜 Coming soon |


## Setup

```bash
# Create and activate the environment
conda create -n isaac python=3.8 -y
conda activate isaac

# Download the code and models (pretrained PHC checkpoints for the SMPL and G1 platforms are provided)

git clone git@github.com:budiu-39/VideoSkills.git
hf download VideoSkills/VideoSkills --local-dir ./VideoSkills

# Install the dependencies
for d in . rsl_rl GVHMR isaacgym/python; do
  (cd "$d" && pip install -e .)
done

pip install -r requirement.txt
```

⚠️ Note: installing `numpy` may print compatibility warnings; they are harmless.



## End-to-end usage (video to robot control)

Run the command below; select the robot platform with `--task` (smpl/g1) and point `--folder` to the directory containing your videos.

```bash

# SMPL robot (2 motion clips in total, takes about 5-10 minutes)
python videoskills/video2agent.py --task=smpl --folder=demo/test_2 --static_cam --headless --accelerate


# G1 robot
python videoskills/video2agent.py --task=g1 --folder=demo/test_2 --static_cam --headless --accelerate
```


**Outputs.** All results are stored under `logs/<smpl/g1>_ppo/<run_name>`:
- `gvhmr_results/`
  Human motion predicted by the motion estimator
- `*.pt`
  Trained robot-control model (checkpoint files)
- `refine_results/`
  Refined (denoised) motion
- `render_results/`
  Rendered comparison videos



## Retargeting (SMPL to G1)

See `scripts/retarget/fit_smpl_motion.py`; it accepts SMPL parameters as input.


## Tracking (zero-shot, with the universal model)

Use the reference motions produced by the retargeting step above, and set `--motion_file=` to the name of their folder (all `.npy` files inside the folder are picked up automatically).


```bash
# G1 robot
python videoskills/eval.py --task=g1 --resume --dev --load_config --load_run=g1_universal --motion_file=AMASS_test

# SMPL robot
python videoskills/eval.py --task=smpl --resume --dev --load_config --load_run=phc_universal --motion_file=AMASS_test
```

To benchmark performance, download the [AMASS test set (for G1)](https://drive.google.com/file/d/1it_7QvfysrSs89h73G2GjYCyVy3-X8Eh/view?usp=sharing) or the [AMASS test set (for SMPL)](https://drive.google.com/file/d/14w_c9ezN3IhkKQT_69GLdsD3FCVJfCPK/view?usp=sharing) and place it in the `dataset` folder.

## Training a PHC model from scratch

Requires motion data that has already been retargeted/preprocessed.

```bash
# G1 robot
python videoskills/train.py --task=g1


# SMPL robot
python videoskills/train.py --task=smpl
```

## Fine-tuning a PHC model on a new dataset

Set `--motion_file=` to the name of the new dataset; add `use_wandb` to enable WandB logging.
```bash
# G1 robot
python videoskills/train.py --task=g1 --resume --load_config --load_run=g1_universal --motion_file=AMASS_test --headless


# SMPL robot
python videoskills/train.py --task=smpl --resume --load_config --load_run=phc_universal --motion_file=AMASS_test --headless
```


## Project layout

```bash
project_root/
├── data/                         # Data and models
│   ├── retarget/                 # Intermediate retargeting data
│   ├── robots/                   # MJCF/XML robot models for MuJoCo
│   └── smpl/                     # SMPL parameters and model files
│
│
├── logs/                         # Training logs, checkpoints, rollouts, renders
│
├── scripts/                      # Data preprocessing and utilities (kept separate from the core code)
│   ├── smpl2sim/                 # Data conversion and preprocessing tools
│   ├── retarget/                 # Retargeting tools
│   ├── render/                   # Rendering and result visualization
│   └── poselib/                  # Motion and skeleton manipulation utilities
│
├── videoskills/                  # Core implementation
│   ├── envs/                     # Task logic and training configs
│   ├── learning/                 # Model architectures (extensions; the core lives in rsl_rl)
│   ├── utils/                    # Utility modules
│   ├── runner/                   # Main training loop
│   └── train.py / video2agent.py # Entry / launcher scripts
│
├── GVHMR/                        # Motion estimation module (GVHMR)
│
├── rsl_rl/                       # Reinforcement learning framework (RSL-RL)
│
└── README.md
```
