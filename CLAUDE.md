# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SeqMultiGrasp is a research codebase for **Sequential Multi-Object Grasping with One Dexterous Hand** (arXiv:2503.09078). It provides a two-stage pipeline: (1) physics-based grasp pose synthesis via simulated annealing optimization, and (2) a diffusion-based generative model that learns to produce grasp poses conditioned on object point clouds.

The robot hand is the **Allegro hand** (right), optionally mounted on a **Franka Panda** arm. Simulation uses **ManiSkill3/SAPIEN**. Motion planning uses **CuRobo**.

## Architecture

### Two Main Modules

**`grasp_generation/`** — Physics-based grasp synthesis and evaluation:
- `main.py`: Entry point. Uses Hydra config (`config/config.yaml`). Runs simulated annealing to optimize hand poses minimizing an energy function (contact force closure + distance + penetration + joint limits).
- `eval.py`: Evaluates merged grasps in ManiSkill3 simulation with CuRobo motion planning.
- `merge.py`: Combines validated single-object grasps into paired HDF5 datasets.
- `visualize_single_object.py` / `visualize_multiple_objects.py`: Plotly-based visualization with tyro CLI.
- `utils/`: Core computation — `hand_model.py` (kinematics, contacts), `object_model.py` (mesh), `energy.py` (optimization objective), `optimizer.py` (simulated annealing), `validation.py` (simulation validation).
- `src/envs/`: ManiSkill3 environments for evaluation (`evaluator/`) and validation (`validator/`).
- `src/motion_planning/`: CuRobo-based trajectory planning.

**`generative_model/`** — Diffusion model for learned grasp generation:
- `train.py`: Training entry point. Hydra config (`config/train.yaml`). DDPM diffusion with wandb/tensorboard logging.
- `inference.py` / `pairwise_object_inference.py`: Generate grasps for single or paired objects.
- `src/network/model.py`: PointNet2 encoder + UNet diffusion backbone with transformer attention.
- `src/data/dataset.py`: Loads paired grasp data from HDF5, samples 1024-point clouds per object.
- `src/refinement.py`: Post-inference grasp refinement.
- Supports multiple rotation representations: `rot6d`, `rotation_matrix`, `quaternion`, `rpy`, `axis_angle`, `keypoints` (configs in `config/model/`).

### Shared Resources

- `robot_models/`: Allegro hand URDF, contact candidates, keypoints metadata.
- `data/meshdata/`: Object meshes (STL/OBJ/URDF) for ~18 object categories.
- `third-party/`: Git submodules — curobo, kaolin, allegro_visualization, pointnet2_ops_lib.

## Key Data Representations

- **Hand state**: 25D vector = 3 (translation) + 6 (rotation, continuous 6D representation) + 16 (joint angles).
- **Grasp dataset (HDF5)**: object names, poses (25D), joint positions (16D), scales, contact point indices.
- **Coordinate convention**: +x right, +y up, +z backward. Palm normal is (1,0,0), up normal is (0,0,1).
- **Mesh caveat**: `.obj` and `.stl` files for the same object differ by a 90° rotation around the x-axis.

## Grasp Validation Pipeline

Progressively stricter filtering: DFC optimization → energy threshold filtering (`*_success.npy`) → floating-hand simulation (`*_validated.npy`) → tabletop simulation with gravity (`*_tabletop_validated.npy`).

## Commands

### Grasp Generation (run from `grasp_generation/`)

```bash
# Single-object grasp synthesis (Hydra CLI)
python main.py object_code=cube name=my_run batch_size=256 n_iter=6000 n_contact=2 \
    'active_links=["link_13.0","link_14.0","link_15.0","link_15.0_tip","link_1.0","link_2.0","link_3.0","link_3.0_tip"]'

# Side grasp variant
python main.py object_code=cylinder_r_2_85_h_10_5 initialization_method=side_grasp ...

# Merge two objects' grasps
python merge.py --path_0 ../data/experiments/.../cube_success_validated.npy \
    --path_1 ../data/experiments/.../cylinder_tabletop_validated.npy \
    --save_path output.h5

# Evaluate in simulation
python eval.py --data_path output.h5 --n 1 --vis

# Visualize
python visualize_single_object.py --data_path .../cube_tabletop_validated.npy --index 0
python visualize_multiple_objects.py --hdf5_path output.h5 --index 0
```

### Generative Model Training (run from `generative_model/`)

```bash
python train.py log_root_dir=logs train.batch_size=512 train.num_epochs=500 \
    data.data_path=<path_to_data.h5> data.object_lists=[obj1,obj2] logger.logger_type=wandb
```

### TensorBoard

```bash
tensorboard --logdir data/experiments/<run_name>/logs
```

## Configuration

- **Grasp generation**: Hydra (`grasp_generation/config/config.yaml`). Key params: `object_code`, `active_links`, `n_contact`, `batch_size`, `n_iter`, `initialization_method` (convex_hull/multi_grasp/side_grasp), energy weights (`w_dis`, `w_pen`, `w_spen`, `w_joints`).
- **Generative model**: Hydra (`generative_model/config/train.yaml`). Key params: `rotation_representation`, `use_keypoints`, model architecture in `config/model/`, scheduler in `config/scheduler/`.

## Tech Stack

- **Python 3.9**, **PyTorch** (2.5.1+), **Hydra** for config, **tyro** for CLI in visualization/eval scripts.
- **PyTorch3D**, **Kaolin** (SDF computation), **CuRobo** (motion planning), **ManiSkill3/SAPIEN** (simulation).
- **Diffusers** (DDPM), **PointNet2** (point cloud encoding via CUDA ops in `third-party/pointnet2_ops_lib`).
- GPU (CUDA) required for all training and most inference.
