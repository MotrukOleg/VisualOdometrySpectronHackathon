# Map-localisation benchmark

Datasets: `crops` (50 task frames), `syn_easy` / `syn_hard` (80-frame synthetic flights over the map with exact GT; hard = screen degradations + strong tilt), `video0061` / `video0062` (own videos, every 5th frame).
Metrics: reg = share of frames registered on the map; err = centre error vs GT (map px); cons = frame-to-frame vs frame-to-map disagreement (GT-free, map px); dev = distance to the SIFT whole-map reference trajectory; fps = localisation speed.

## Methods

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orb_bf_global ¹ | 100% | 0.54 | 100% | 0.79 | 82% | 100% | 100% | 0.301 | 0.39 | 0% | 1.1 | 0 |
| superpoint_lightglue_track ¹ | 100% | 0.61 | 100% | 0.16 | 100% | 100% | 100% | 0.214 | 0.47 | 0% | 4.0 | 6089 |
| superpoint_superglue_official_track ¹ | 100% | 0.63 | 100% | 0.13 | 100% | 100% | 100% | 0.333 | 0.72 | 0% | 1.4 | 1113 |
| sift_flann_track ¹ | 100% | 0.65 | 100% | 0.03 | 100% | 100% | 100% | 0.080 | 0.32 | 0% | 2.9 | 9 |
| efficient_loftr_official_track ¹ | 100% | 0.65 | 100% | 0.19 | 100% | 100% | 100% | 0.312 | 0.52 | 10% | 1.7 | 3830 |
| sift_lightglue_track ¹ | 100% | 0.66 | 100% | 0.06 | 100% | 100% | 100% | 0.196 | 0.29 | 0% | 5.0 | 192 |
| akaze_bf_track ¹ | 100% | 0.67 | 100% | 0.03 | 100% | 100% | 100% | 0.085 | 0.31 | 0% | 7.2 | 0 |
| disk_lightglue_track ¹ | 100% | 0.67 | 100% | 0.07 | 100% | 100% | 100% | 0.108 | 0.66 | 0% | 3.2 | 12264 |
| sift_adalam_track ¹ | 100% | 0.67 | 100% | 0.03 | 100% | 100% | 100% | 0.065 | 0.32 | 0% | 3.7 | 222 |
| loftr_track ¹ | 100% | 0.68 | 100% | 0.14 | 100% | 100% | 100% | 0.358 | 0.61 | 4% | 2.9 | 1499 |
| orb_bf_track ¹ | 100% | 0.69 | 100% | 0.16 | 100% | 100% | 100% | 0.281 | 0.39 | 0% | 6.9 | 9 |
| xfeat_track ¹ | 100% | 0.70 | 100% | 0.40 | 100% | 100% | 100% | 0.535 | 0.59 | 22% | 11.2 | 404 |
| rootsift_flann_global ¹ | 100% | 0.71 | 100% | 0.35 | 100% | 100% | 100% | 0.048 | 0.03 | 0% | 8.6 | 0 |
| map_baseline_sift | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.045 | – | 0% | 8.3 | 0 |
| sift_mnn_global ¹ | 100% | 0.72 | 100% | 0.35 | 90% | 100% | 100% | 0.049 | 0.05 | 0% | 1.0 | 0 |
| sift_flann_global ¹ | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.050 | 0.03 | 0% | 8.5 | 0 |
| aliked_n16rot_mnn_track ¹ | 100% | 0.75 | 100% | 0.35 | 100% | 100% | 100% | 0.086 | 0.52 | 82% | 4.8 | 10379 |
| aliked_lightglue_track ¹ | 100% | 0.77 | 100% | 0.22 | 100% | 100% | 100% | 0.119 | 0.94 | 0% | 3.3 | 9428 |
| superpoint_superglue_track ¹ | 100% | 0.88 | 100% | 0.23 | 100% | 100% | 100% | 0.389 | 1.19 | 0% | 1.8 | 690 |
| xfeat_star_track ¹ | 100% | 1.45 | 100% | 1.34 | 100% | 100% | 100% | 0.458 | 1.86 | 11% | 8.1 | 657 |
| brisk_bf_global ¹ | 98% | 0.30 | 95% | 0.43 | 98% | 100% | 100% | 0.172 | 0.38 | 0% | 4.3 | 0 |
| akaze_bf_global ¹ | 98% | 0.31 | 92% | 0.19 | 94% | 100% | 100% | 0.110 | 0.25 | 0% | 6.2 | 0 |
| roma_track ¹ | 96% | 0.70 | 92% | 0.05 | 100% | – | 100% | 0.342 | 0.59 | 0% | 0.1 | 4384 |
| dedode_v2_mnn_track ¹ | 96% | 0.78 | 92% | 0.15 | 100% | 100% | 100% | 0.282 | 0.57 | 4% | 0.8 | 10455 |
| tiny_roma_track ¹ | 94% | 0.92 | 100% | 0.98 | 100% | 100% | 100% | 1.470 | 0.85 | 28% | 3.6 | 602 |


¹ videos evaluated on every 2nd keyframe (`…s2`, every 10th video frame) to fit the time budget. SIFT fallback = share of frames located by the global SIFT re-initialisation instead of the method itself; a high value means the method mostly failed and its numbers are SIFT's.


Excluded from the ranking:
- `efficient_loftr_track`: transformers port: coarse-only matches (8-px grid, no sub-pixel refinement) and a fixed 640x480 resize of both images; see efficient_loftr_official_track

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| efficient_loftr_track ¹ | 64% | 1.04 | 94% | 6.19 | 100% | 100% | 100% | 20.965 | 6.85 | 57% | 2.0 | 1011 |


## Cross-matcher consistency (GT-free, family-neutral)

Median disagreement (map px) between consecutive map registrations and frame-to-frame homographies estimated by matchers of OTHER families than the method (SIFT / SuperPoint+LightGlue / LoFTR). Videos on every 2nd keyframe. Lower = smoother; it does not measure absolute accuracy (Spearman correlation with the true error over methods on the synthetic sets is only syn_easy: 0.43, syn_hard: 0.24).

| method | crops | syn_easy | syn_hard | video0061 | video0062 | video214816 |
|---|---|---|---|---|---|---|
| aliked_n16rot_mnn_track | 0.125 | 0.167 | 0.208 | 0.086 | 0.139 | 0.300 |
| akaze_bf_track | 0.169 | 0.193 | 0.211 | 0.079 | 0.118 | 0.263 |
| map_baseline_sift | 0.149 | 0.195 | 0.232 | 0.075 | 0.115 | 0.271 |
| rootsift_flann_global | 0.149 | 0.192 | 0.239 | 0.077 | 0.131 | 0.264 |
| sift_mnn_global | 0.148 | 0.191 | 0.232 | 0.073 | 0.109 | 0.301 |
| sift_flann_global | 0.148 | 0.196 | 0.244 | 0.074 | 0.123 | 0.275 |
| sift_flann_track | 0.174 | 0.210 | 0.257 | 0.082 | 0.151 | 0.294 |
| sift_adalam_track | 0.188 | 0.207 | 0.259 | 0.080 | 0.156 | 0.295 |
| disk_lightglue_track | 0.218 | 0.208 | 0.262 | 0.086 | 0.142 | 0.289 |
| aliked_lightglue_track | 0.220 | 0.279 | 0.346 | 0.091 | 0.123 | 0.337 |
| akaze_bf_global | 0.293 | 0.366 | 0.351 | 0.062 | 0.088 | 0.267 |
| brisk_bf_global | 0.372 | 0.441 | 0.302 | 0.087 | 0.124 | 0.283 |
| superpoint_lightglue_track | 0.405 | 0.414 | 0.244 | 0.104 | 0.132 | 0.347 |
| sift_lightglue_track | 0.378 | 0.276 | 0.356 | 0.138 | 0.194 | 0.322 |
| roma_track | 0.112 | 0.174 | 0.306 | – | 0.635 | – |
| orb_bf_track | 0.342 | 0.373 | 0.372 | 0.267 | 0.300 | 0.326 |
| dedode_v2_mnn_track | 0.235 | 0.267 | 0.510 | 0.344 | 0.342 | – |
| efficient_loftr_official_track | 0.377 | 0.314 | 0.305 | 0.270 | 0.357 | 0.580 |
| superpoint_superglue_official_track | 0.298 | 0.342 | 0.245 | 0.330 | 0.397 | 0.660 |
| orb_bf_global | 0.797 | 0.832 | 0.312 | 0.086 | 0.143 | 0.232 |
| loftr_track | 0.256 | 0.282 | 0.343 | 0.408 | 0.473 | 0.694 |
| superpoint_superglue_track | 0.270 | 0.367 | 0.313 | 0.469 | 0.451 | 0.786 |
| xfeat_star_track | 1.089 | 1.342 | 0.500 | 0.204 | 0.163 | 0.304 |
| xfeat_track | 1.129 | 1.321 | 0.462 | 0.241 | 0.221 | 0.357 |
| tiny_roma_track | 3.288 | 3.415 | 1.110 | 0.513 | 0.623 | 0.865 |


## New video `video214816` (strong roll + perspective; every 2nd keyframe)

| method | reg | cons med, px | cons p95, px | dev vs SIFT med, px | dev p95, px | SIFT fallback | inliers | fps |
|---|---|---|---|---|---|---|---|---|
| sift_flann_global ¹ | 100% | 0.158 | 0.44 | 0.03 | 0.24 | 0% | 1594 | 5.82 |
| rootsift_flann_global ¹ | 100% | 0.161 | 0.41 | 0.04 | 0.24 | 0% | 1657 | 5.69 |
| sift_mnn_global ¹ | 100% | 0.178 | 0.43 | 0.05 | 0.27 | 0% | 1970 | 0.95 |
| sift_flann_track ¹ | 100% | 0.187 | 0.52 | 0.36 | 0.60 | 0% | 300 | 1.26 |
| akaze_bf_track ¹ | 100% | 0.191 | 0.50 | 0.36 | 0.66 | 0% | 581 | 0.95 |
| sift_lightglue_track ¹ | 100% | 0.203 | 0.59 | 0.38 | 0.72 | 0% | 209 | 0.86 |
| akaze_bf_global ¹ | 100% | 0.193 | 0.47 | 0.38 | 1.09 | 0% | 544 | 1.54 |
| sift_adalam_track ¹ | 100% | 0.194 | 0.52 | 0.38 | 0.61 | 0% | 395 | 1.31 |
| orb_bf_global ¹ | 100% | 0.171 | 0.39 | 0.41 | 1.90 | 0% | 1054 | 3.49 |
| orb_bf_track ¹ | 100% | 0.266 | 0.63 | 0.41 | 0.72 | 0% | 324 | 3.43 |
| brisk_bf_global ¹ | 100% | 0.201 | 0.48 | 0.42 | 1.02 | 0% | 648 | 1.17 |
| disk_lightglue_track ¹ | 100% | 0.225 | 0.58 | 0.53 | 0.89 | 0% | 384 | 0.29 |
| aliked_n16rot_mnn_track ¹ | 100% | 0.237 | 0.57 | 0.59 | 0.97 | 9% | 512 | 0.40 |
| xfeat_track ¹ | 100% | 0.306 | 0.74 | 0.63 | 1.12 | 0% | 494 | 2.63 |
| superpoint_lightglue_track ¹ | 100% | 0.359 | 0.75 | 0.67 | 1.12 | 0% | 328 | 0.37 |
| efficient_loftr_official_track ¹ | 100% | 0.544 | 1.54 | 0.85 | 1.40 | 0% | 2651 | 0.65 |
| aliked_lightglue_track ¹ | 100% | 0.262 | 0.72 | 0.86 | 1.30 | 0% | 415 | 0.32 |
| loftr_track ¹ | 100% | 0.662 | 1.66 | 0.93 | 1.66 | 0% | 1517 | 0.73 |
| tiny_roma_track ¹ | 100% | 0.854 | 2.23 | 1.20 | 2.06 | 0% | 1553 | 3.36 |
| superpoint_superglue_official_track ¹ | 100% | 0.620 | 1.50 | 1.29 | 2.02 | 0% | 450 | 0.62 |
| xfeat_star_track ¹ | 100% | 0.232 | 0.70 | 2.00 | 2.54 | 0% | 1370 | 1.24 |
| superpoint_superglue_track ¹ | 100% | 0.772 | 1.67 | 2.22 | 3.13 | 0% | 191 | 1.31 |
| map_baseline_sift | 100% | 0.110 | 0.31 | – | – | 0% | 1593 | 5.55 |


## ablation_track_sift

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sift_flann_track (base) ¹ | 100% | 0.65 | 100% | 0.03 | 100% | 100% | 100% | 0.080 | 0.32 | 0% | 2.9 | 9 |
| sift_flann_track[tracking.roi_factor=3.0] | 96% | 0.66 | 92% | 0.04 | 100% | 100% | 100% | 0.090 | 0.31 | 2% | 2.1 | 0 |
| sift_flann_track[tracking.motion_model=static] | 96% | 0.67 | 92% | 0.03 | 100% | 100% | 100% | 0.084 | 0.33 | 0% | 3.4 | 0 |
| sift_flann_track[tracking.auto_scale=false] | 96% | 0.69 | 92% | 0.18 | 100% | 100% | 100% | 0.063 | 0.35 | 0% | 4.9 | 0 |
| sift_flann_track[tracking.roi_factor=1.2] | 96% | 0.67 | 92% | 0.04 | 100% | 100% | 100% | 0.076 | 0.32 | 0% | 3.8 | 0 |


## rotation

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sift_flann_track (base) ¹ | 100% | 0.65 | 100% | 0.03 | 100% | 100% | 100% | 0.080 | 0.32 | 0% | 2.9 | 9 |
| superpoint_lightglue_track (base) ¹ | 100% | 0.61 | 100% | 0.16 | 100% | 100% | 100% | 0.214 | 0.47 | 0% | 4.0 | 6089 |
| loftr_track (base) ¹ | 100% | 0.68 | 100% | 0.14 | 100% | 100% | 100% | 0.358 | 0.61 | 4% | 2.9 | 1499 |
| sift_flann_track[tracking.rotation=test] | 96% | 0.87 | 92% | 0.55 | 100% | 100% | 100% | 0.500 | 0.36 | 0% | 1.1 | 0 |
| superpoint_lightglue_track[tracking.rotation=test] ¹ | 96% | 0.69 | 92% | – | 100% | – | 100% | 0.342 | 0.48 | 4% | 1.7 | 4516 |
| superpoint_lightglue_track[tracking.rotation=none] ¹ | 96% | 0.75 | 92% | – | 100% | – | 100% | 0.088 | 0.43 | 84% | 5.6 | 4516 |
| loftr_track[tracking.rotation=test] ¹ | 95% | 0.77 | 95% | – | 100% | – | 100% | 0.365 | 0.49 | 14% | 1.1 | 1466 |
| sift_flann_track[tracking.rotation=none] | 96% | 1.02 | 92% | 0.52 | 100% | 100% | 100% | 0.079 | 0.33 | 0% | 4.3 | 0 |
| loftr_track[tracking.rotation=none] ¹ | 96% | 0.78 | 92% | – | 100% | – | 100% | 0.158 | 0.25 | 82% | 4.3 | 1401 |


## postprocess

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| map_baseline_sift (base) | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.045 | – | 0% | 8.3 | 0 |
| map_baseline_sift[smoothing.savgol_window=15] | 21% | 11.50 | 92% | 17.84 | 100% | 100% | 100% | 5.324 | 6.80 | 0% | 7.7 | 0 |
| map_baseline_sift[smoothing.savgol_window=7] | 95% | 1.02 | 92% | 0.93 | 100% | 100% | 100% | 4.155 | 3.24 | 0% | 8.3 | 0 |
| map_baseline_sift[smoothing.interpolate=false] | 92% | 0.69 | 92% | 0.35 | 100% | 100% | 100% | 0.046 | 0.02 | 0% | 8.7 | 0 |


## ablation_track_splg

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| superpoint_lightglue_track (base) ¹ | 100% | 0.61 | 100% | 0.16 | 100% | 100% | 100% | 0.214 | 0.47 | 0% | 4.0 | 6089 |
| superpoint_lightglue_track[tracking.auto_scale=false] ¹ | 96% | 0.66 | 92% | – | 100% | – | 100% | 0.319 | 0.43 | 2% | 3.5 | 4517 |
| superpoint_lightglue_track[tracking.work_scale=2.0] ¹ | 96% | 0.73 | 92% | – | 100% | – | 100% | 0.214 | 0.22 | 1% | 1.3 | 12842 |
| superpoint_lightglue_track[tracking.motion_model=static] ¹ | 96% | 0.63 | 92% | – | 100% | – | 100% | 0.306 | 0.47 | 1% | 5.0 | 4517 |
| superpoint_lightglue_track[preprocess.blur_sigma=1.0] ¹ | 96% | 0.64 | 92% | – | 100% | – | 100% | 0.260 | 0.45 | 1% | 5.0 | 4517 |
| superpoint_lightglue_track[geometry.estimator=ransac] ¹ | 96% | 0.66 | 92% | – | 100% | – | 100% | 0.360 | 0.51 | 1% | 5.0 | 4528 |
| superpoint_lightglue_track[preprocess.clahe=true] ¹ | 96% | 0.64 | 92% | – | 100% | – | 100% | 0.256 | 0.46 | 1% | 4.7 | 4521 |
| superpoint_lightglue_track[preprocess.screen_mask=true] ¹ | 96% | 0.64 | 92% | – | 100% | – | 100% | 0.301 | 0.48 | 4% | 4.9 | 4521 |


## search

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| map_baseline_sift (base) | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.045 | – | 0% | 8.3 | 0 |
| map_baseline_sift[localize.mode=tiles] | 96% | 1.02 | 92% | 0.51 | 100% | 100% | 98% | 0.089 | 0.27 | 0% | 0.2 | 0 |


## areas

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| superpoint_lightglue_track (base) ¹ | 100% | 0.61 | 100% | 0.16 | 100% | 100% | 100% | 0.214 | 0.47 | 0% | 4.0 | 6089 |
| superpoint_superglue_official_track (base) ¹ | 100% | 0.63 | 100% | 0.13 | 100% | 100% | 100% | 0.333 | 0.72 | 0% | 1.4 | 1113 |
| sift_lightglue_track (base) ¹ | 100% | 0.66 | 100% | 0.06 | 100% | 100% | 100% | 0.196 | 0.29 | 0% | 5.0 | 192 |
| xfeat_track (base) ¹ | 100% | 0.70 | 100% | 0.40 | 100% | 100% | 100% | 0.535 | 0.59 | 22% | 11.2 | 404 |
| orb_bf_track (base) ¹ | 100% | 0.69 | 100% | 0.16 | 100% | 100% | 100% | 0.281 | 0.39 | 0% | 6.9 | 9 |
| efficient_loftr_official_track (base) ¹ | 100% | 0.65 | 100% | 0.19 | 100% | 100% | 100% | 0.312 | 0.52 | 10% | 1.7 | 3830 |
| sift_flann_track (base) ¹ | 100% | 0.65 | 100% | 0.03 | 100% | 100% | 100% | 0.080 | 0.32 | 0% | 2.9 | 9 |
| akaze_bf_track (base) ¹ | 100% | 0.67 | 100% | 0.03 | 100% | 100% | 100% | 0.085 | 0.31 | 0% | 7.2 | 0 |
| sift_adalam_track (base) ¹ | 100% | 0.67 | 100% | 0.03 | 100% | 100% | 100% | 0.065 | 0.32 | 0% | 3.7 | 222 |
| loftr_track (base) ¹ | 100% | 0.68 | 100% | 0.14 | 100% | 100% | 100% | 0.358 | 0.61 | 4% | 2.9 | 1499 |
| superpoint_lightglue_track | – | – | – | – | – | – | – | – | – | 50% | 9.8 | 332 |
| superpoint_superglue_official_track | – | – | – | – | – | – | – | – | – | 60% | 11.1 | 326 |
| sift_lightglue_track | – | – | – | – | – | – | – | – | – | 2% | 7.0 | 92 |
| xfeat_track | – | – | – | – | – | – | – | – | – | 58% | 18.3 | 167 |
| orb_bf_track | – | – | – | – | – | – | – | – | – | 1% | 24.4 | 8 |
| efficient_loftr_official_track | – | – | – | – | – | – | – | – | – | 48% | 11.8 | 778 |
| sift_flann_track | – | – | – | – | – | – | – | – | – | 2% | 22.6 | 0 |
| akaze_bf_track | – | – | – | – | – | – | – | – | – | 22% | 29.6 | 0 |
| sift_adalam_track | – | – | – | – | – | – | – | – | – | 4% | 21.0 | 19 |
| loftr_track | – | – | – | – | – | – | – | – | – | 12% | 7.4 | 844 |


## preprocess

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| map_baseline_sift (base) | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.045 | – | 0% | 8.3 | 0 |
| map_baseline_sift[preprocess.screen_mask=true] | 96% | 0.71 | 92% | 0.35 | 100% | 100% | 100% | 0.046 | 0.03 | 0% | 9.5 | 0 |
| map_baseline_sift[preprocess.blur_sigma=2.0] | 95% | 0.78 | 86% | 0.36 | 100% | 100% | 100% | 0.078 | 0.12 | 0% | 20.4 | 0 |
| map_baseline_sift[preprocess.clahe=true] | 96% | 0.73 | 92% | 0.35 | 100% | 100% | 100% | 0.044 | 0.10 | 0% | 7.1 | 0 |
| map_baseline_sift[preprocess.blur_sigma=1.0] | 96% | 0.73 | 92% | 0.35 | 100% | 100% | 100% | 0.051 | 0.07 | 0% | 12.7 | 0 |


## estimators

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| map_baseline_sift (base) | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.045 | – | 0% | 8.3 | 0 |
| map_baseline_sift[geometry.estimator=usac_accurate,localize.reproj_threshold=5] | 96% | 0.73 | 92% | 0.35 | 100% | 100% | 100% | 0.043 | 0.02 | 0% | 9.7 | 0 |
| map_baseline_sift[geometry.estimator=ransac,localize.reproj_threshold=5] | 96% | 0.71 | 92% | 0.36 | 100% | 100% | 100% | 0.054 | 0.04 | 0% | 9.8 | 0 |
| map_baseline_sift[geometry.estimator=magsac,localize.reproj_threshold=2] | 96% | 0.73 | 92% | 0.36 | 100% | 100% | 100% | 0.053 | 0.04 | 0% | 9.8 | 0 |
| map_baseline_sift[geometry.estimator=magsac,localize.reproj_threshold=8] | 96% | 0.72 | 92% | 0.35 | 100% | 100% | 100% | 0.045 | 0.02 | 0% | 9.7 | 0 |
| map_baseline_sift[geometry.estimator=ransac,localize.reproj_threshold=2] | 96% | 0.74 | 92% | 0.35 | 100% | 100% | 100% | 0.073 | 0.09 | 0% | 9.8 | 0 |
| map_baseline_sift[geometry.estimator=magsac,localize.reproj_threshold=3] | 96% | 0.71 | 92% | 0.36 | 100% | 100% | 100% | 0.053 | 0.03 | 0% | 9.8 | 0 |


## frame_scale

| method | syn_hard ≤5px | syn_hard err med, px | syn_hard reg | syn_easy err med, px | crops reg | v61 reg | v62 reg | real cons, px | v62 dev vs SIFT, px | SIFT fallback (max) | fps (mean) | VRAM, MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| map_baseline_sift (base) | 100% | 0.72 | 100% | 0.35 | 100% | 100% | 100% | 0.045 | – | 0% | 8.3 | 0 |
| map_baseline_sift[localize.frame_scale=0.35] | 96% | 0.59 | 92% | 0.35 | 100% | 100% | 100% | 0.045 | 0.17 | 0% | 13.7 | 0 |
| map_baseline_sift[localize.frame_scale=0.25] | 96% | 0.46 | 92% | 0.37 | 100% | 100% | 100% | 0.049 | 0.38 | 0% | 22.0 | 0 |
| map_baseline_sift[localize.frame_scale=1.0] | 96% | 0.91 | 92% | 0.38 | 100% | 100% | 100% | 0.049 | 0.20 | 0% | 5.2 | 0 |
