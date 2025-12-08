#!/bin/bash
# Test with 3 DCs in Daejeon area
# DC locations based on GPKG bounding box (36.27-36.49, 127.25-127.41):
# - DC1 (North):  36.448655, 127.329297
# - DC2 (Center): 36.380128, 127.329297
# - DC3 (South):  36.311600, 127.345514

python -m src.vrp_fairness.run_experiment \
  --seed 0 \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --housing-type 공동주택 \
  --dcs "36.448655,127.329297" "36.380128,127.329297" "36.311600,127.345514" \
  --eps 0.10 \
  --iters 300
