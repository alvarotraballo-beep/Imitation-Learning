#!/usr/bin/env bash
set -euo pipefail

cd "/home/alvaro/projects/Imitation Learning JAKA"

robomimic_env/bin/python generate_bag_direct_ctrl_dataset.py \
  --output tests/assets/bag_lift_direct_ctrl_basetwist30_openfix_variants_allbags_3var.hdf5 \
  --max-bags 0 \
  --cube-variants 3 \
  --cube-pos-jitter 0.022 \
  --variant-sampling halton \
  --cube-size-values "0.018,0.020,0.021" \
  --canonical-cube-xy "0.0144,-0.0123" \
  --waypoint-style bag_center_smooth \
  --grasp-cube-z-offset -0.025 \
  --base-twist-deg 30 \
  --ik-seed-q "0.6981,0.2113,-0.9254,0.0464,-0.0803,1.0472" \
  --wrist-q6 1.0472 \
  --steps-per-path 8 \
  --close-steps 60 \
  --lift-steps 24 \
  --gripper-boost-force 650 \
  --grasp-friction 12
