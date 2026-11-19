#!/usr/bin/env bash
# Classical part of the benchmark (no downloads): methods, estimators, scale, preprocessing, search, rotation, post.
set -u
P=${PY:-/e/Anaconda/envs/MachineLearning/python.exe}
B="$P scripts/run_map_benchmark.py --no-viz"
M=configs/map/methods
$B --tag methods --configs $M/sift_flann.yaml $M/rootsift_flann.yaml $M/sift_mnn.yaml $M/orb_bf.yaml $M/akaze_bf.yaml $M/brisk_bf.yaml
$B --tag methods --configs $M/track_sift.yaml $M/track_sift_adalam.yaml $M/track_orb.yaml $M/track_akaze.yaml
$B --tag estimators --configs configs/map/baseline_sift.yaml --variants \
   "geometry.estimator=ransac,localize.reproj_threshold=5" "geometry.estimator=usac_accurate,localize.reproj_threshold=5" \
   "geometry.estimator=magsac,localize.reproj_threshold=2" "geometry.estimator=magsac,localize.reproj_threshold=3" \
   "geometry.estimator=magsac,localize.reproj_threshold=8" "geometry.estimator=ransac,localize.reproj_threshold=2"
$B --tag frame_scale --configs configs/map/baseline_sift.yaml --variants \
   "localize.frame_scale=1.0" "localize.frame_scale=0.35" "localize.frame_scale=0.25"
$B --tag preprocess --configs configs/map/baseline_sift.yaml --variants \
   "preprocess.clahe=true" "preprocess.blur_sigma=1.0" "preprocess.blur_sigma=2.0" "preprocess.screen_mask=true"
$B --tag search --configs configs/map/baseline_sift.yaml --variants "localize.mode=tiles"
$B --tag rotation --configs $M/track_sift.yaml --variants "tracking.rotation=none" "tracking.rotation=test"
$B --tag ablation_track_sift --configs $M/track_sift.yaml --variants \
   "tracking.auto_scale=false" "tracking.motion_model=static" "tracking.roi_factor=3.0" "tracking.roi_factor=1.2"
$B --tag postprocess --configs configs/map/baseline_sift.yaml --variants \
   "smoothing.interpolate=false" "smoothing.savgol_window=7" "smoothing.savgol_window=15"
echo CLASSICAL_DONE
