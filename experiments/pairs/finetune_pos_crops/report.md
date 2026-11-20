# Pair benchmark with exact ground truth (`data/_finetune_pos_crops`)

Consecutive crops of one north-up orthophoto; the true shift is found by an exhaustive pixel-exact search (no feature matcher involved). Sequence 11 has no overlapping consecutive crops and is excluded. Error = mean corner error of img1 (512 px), failed estimate = ∞. One estimator for all methods (homography, MAGSAC++, 3 px). Variants: `plain` as is (ortho vs ortho, shift); `geo` + rotation ±15°, scale 0.75–1.3; `hard` + screen degradations.

Pairs per variant (common to all methods): plain 210, geo 210, hard 210. ms = median time per pair; `gpu` = RTX 5060 Ti server (shared), `local` = GTX 1650 dev PC — times are comparable only within one machine.

| method | plain @1px | plain @3px | plain med px | geo @1px | geo @3px | geo med px | hard @1px | hard @3px | hard med px | pairs | ms (plain) | machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sift_lightglue_track | 92% | 99% | 0.16 | 71% | 88% | 0.49 | 10% | 39% | 4.72 | 210 | 300 | local |
| efficient_loftr_official_track | 96% | 100% | 0.12 | 71% | 96% | 0.59 | 5% | 34% | 6.26 | 210 | 2402 | gpu |
| aliked_lightglue_track | 72% | 98% | 0.54 | 50% | 83% | 1.02 | 4% | 34% | 6.51 | 210 | 223 | gpu |
| disk_lightglue_track | 73% | 96% | 0.42 | 51% | 78% | 0.93 | 4% | 30% | 6.97 | 210 | 267 | gpu |
| loftr_track | 90% | 100% | 0.26 | 80% | 97% | 0.41 | 6% | 28% | 8.26 | 210 | 2880 | local |
| xfeat_track | 43% | 72% | 1.26 | 24% | 54% | 2.48 | 1% | 25% | 17.91 | 210 | 27 | local |
| xfeat_star_track | 8% | 78% | 1.84 | 4% | 67% | 2.16 | 0% | 23% | 7.75 | 210 | 64 | local |
| aliked_n16rot_mnn_track | 62% | 90% | 0.71 | 47% | 80% | 1.07 | 5% | 22% | 11.64 | 210 | 695 | gpu |
| sift_mnn_global | 87% | 94% | 0.13 | 79% | 87% | 0.27 | 2% | 20% | inf | 210 | 110 | gpu |
| orb_bf_track | 52% | 78% | 0.95 | 35% | 62% | 1.79 | 0% | 20% | 22.22 | 210 | 69 | local |
| superpoint_lightglue_track | 42% | 78% | 1.16 | 26% | 68% | 1.72 | 0% | 19% | 11.51 | 210 | 83 | local |
| sift_flann_track | 89% | 95% | 0.11 | 81% | 89% | 0.20 | 1% | 19% | inf | 210 | 172 | local |
| sift_adalam_track | 88% | 91% | 0.10 | 80% | 87% | 0.22 | 2% | 19% | inf | 210 | 166 | local |
| rootsift_flann_global | 90% | 96% | 0.10 | 80% | 89% | 0.19 | 2% | 18% | inf | 210 | 154 | gpu |
| tiny_roma_track | 20% | 59% | 2.05 | 6% | 44% | 3.66 | 0% | 14% | 11.28 | 210 | 79 | local |
| superpoint_superglue_official_track | 47% | 75% | 1.11 | 36% | 65% | 1.63 | 0% | 14% | 18.33 | 210 | 108 | local |
| superpoint_superglue_track | 46% | 80% | 1.10 | 32% | 64% | 1.80 | 1% | 12% | 20.34 | 210 | 142 | local |
| brisk_bf_global | 59% | 73% | 0.68 | 39% | 63% | 1.56 | 0% | 7% | inf | 210 | 132 | gpu |
| akaze_bf_track | 64% | 71% | 0.34 | 35% | 51% | 2.58 | 0% | 5% | inf | 210 | 43 | gpu |
| efficient_loftr_track ² | 1% | 12% | 51.18 | 0% | 1% | 214.13 | 0% | 0% | inf | 210 | 293 | local |


² efficient_loftr_track: transformers port (coarse-only matches)


## Slow methods on a 120-pair selection (references on exactly the same views)

| method | plain @1px | plain @3px | plain med px | geo @1px | geo @3px | geo med px | hard @1px | hard @3px | hard med px | pairs | ms (plain) | machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dedode_v2_mnn_track | 100% | 100% | 0.00 | 64% | 95% | 0.68 | 4% | 38% | 5.42 | 120 | 10890 | gpu |
| efficient_loftr_official_track | 94% | 99% | 0.12 | 77% | 97% | 0.55 | 8% | 37% | 5.62 | 120 | 188 | gpu |
| sift_lightglue_track | 95% | 99% | 0.16 | 69% | 91% | 0.46 | 8% | 37% | 4.45 | 120 | 230 | local |
| disk_lightglue_track | 74% | 91% | 0.36 | 53% | 82% | 0.90 | 5% | 31% | 7.02 | 120 | 307 | gpu |
| loftr_track | 88% | 98% | 0.28 | 80% | 98% | 0.38 | 6% | 29% | 7.01 | 120 | 345 | local |
| sift_flann_track | 92% | 96% | 0.09 | 82% | 90% | 0.20 | 3% | 22% | inf | 120 | 180 | local |
| superpoint_lightglue_track | 42% | 82% | 1.27 | 31% | 70% | 1.65 | 2% | 18% | 9.45 | 120 | 85 | local |


## Success @3px by terrain, variant `plain`

| method | 01 suburb | 02 fields | 03 town + river | 04 industrial / wetland | 05 fields + forest | 08 water + fields |
|---|---|---|---|---|---|---|
| sift_lightglue_track | 100% | 100% | 97% | 100% | 100% | 97% |
| efficient_loftr_official_track | 100% | 100% | 100% | 97% | 100% | 100% |
| aliked_lightglue_track | 100% | 100% | 100% | 91% | 100% | 94% |
| disk_lightglue_track | 100% | 97% | 100% | 86% | 100% | 91% |
| loftr_track | 100% | 100% | 100% | 100% | 100% | 100% |
| xfeat_track | 100% | 100% | 69% | 54% | 94% | 17% |
| xfeat_star_track | 100% | 91% | 74% | 51% | 100% | 51% |
| aliked_n16rot_mnn_track | 97% | 91% | 94% | 74% | 100% | 83% |
| sift_mnn_global | 97% | 91% | 100% | 80% | 100% | 97% |
| orb_bf_track | 97% | 89% | 63% | 51% | 100% | 66% |
| superpoint_lightglue_track | 100% | 100% | 69% | 49% | 100% | 49% |
| sift_flann_track | 97% | 94% | 100% | 80% | 100% | 100% |
| sift_adalam_track | 97% | 89% | 100% | 66% | 100% | 97% |
| rootsift_flann_global | 97% | 94% | 100% | 83% | 100% | 100% |
| tiny_roma_track | 91% | 86% | 40% | 26% | 97% | 14% |
| superpoint_superglue_official_track | 100% | 89% | 69% | 49% | 100% | 43% |
| superpoint_superglue_track | 97% | 89% | 83% | 54% | 100% | 60% |
| brisk_bf_global | 91% | 89% | 57% | 20% | 100% | 83% |
| akaze_bf_track | 89% | 86% | 60% | 29% | 100% | 66% |
| efficient_loftr_track | 6% | 9% | 3% | 0% | 54% | 0% |


## Success @3px by terrain, variant `geo`

| method | 01 suburb | 02 fields | 03 town + river | 04 industrial / wetland | 05 fields + forest | 08 water + fields |
|---|---|---|---|---|---|---|
| sift_lightglue_track | 100% | 100% | 86% | 71% | 100% | 69% |
| efficient_loftr_official_track | 100% | 100% | 91% | 94% | 100% | 91% |
| aliked_lightglue_track | 97% | 94% | 74% | 57% | 100% | 77% |
| disk_lightglue_track | 94% | 97% | 71% | 46% | 100% | 60% |
| loftr_track | 100% | 100% | 89% | 97% | 100% | 97% |
| xfeat_track | 94% | 91% | 20% | 14% | 97% | 9% |
| xfeat_star_track | 97% | 89% | 54% | 34% | 97% | 29% |
| aliked_n16rot_mnn_track | 97% | 89% | 71% | 57% | 97% | 69% |
| sift_mnn_global | 97% | 94% | 86% | 57% | 100% | 86% |
| orb_bf_track | 91% | 86% | 49% | 20% | 100% | 26% |
| superpoint_lightglue_track | 97% | 86% | 63% | 31% | 100% | 31% |
| sift_flann_track | 97% | 91% | 89% | 69% | 100% | 89% |
| sift_adalam_track | 97% | 91% | 89% | 57% | 100% | 89% |
| rootsift_flann_global | 97% | 94% | 89% | 66% | 100% | 89% |
| tiny_roma_track | 66% | 63% | 26% | 9% | 89% | 11% |
| superpoint_superglue_official_track | 97% | 86% | 49% | 29% | 97% | 31% |
| superpoint_superglue_track | 94% | 83% | 57% | 26% | 97% | 29% |
| brisk_bf_global | 89% | 86% | 40% | 20% | 97% | 49% |
| akaze_bf_track | 83% | 74% | 20% | 14% | 97% | 20% |
| efficient_loftr_track | 3% | 0% | 0% | 0% | 3% | 0% |


## Success @3px by terrain, variant `hard`

| method | 01 suburb | 02 fields | 03 town + river | 04 industrial / wetland | 05 fields + forest | 08 water + fields |
|---|---|---|---|---|---|---|
| sift_lightglue_track | 63% | 51% | 17% | 11% | 80% | 11% |
| efficient_loftr_official_track | 51% | 57% | 14% | 6% | 69% | 9% |
| aliked_lightglue_track | 60% | 51% | 14% | 9% | 60% | 9% |
| disk_lightglue_track | 54% | 51% | 14% | 0% | 54% | 6% |
| loftr_track | 40% | 46% | 11% | 3% | 60% | 9% |
| xfeat_track | 57% | 34% | 0% | 3% | 54% | 0% |
| xfeat_star_track | 40% | 40% | 6% | 0% | 54% | 0% |
| aliked_n16rot_mnn_track | 37% | 40% | 6% | 0% | 43% | 6% |
| sift_mnn_global | 57% | 34% | 3% | 3% | 17% | 6% |
| orb_bf_track | 40% | 34% | 0% | 0% | 46% | 0% |
| superpoint_lightglue_track | 37% | 20% | 0% | 0% | 49% | 6% |
| sift_flann_track | 49% | 40% | 0% | 9% | 9% | 6% |
| sift_adalam_track | 54% | 37% | 0% | 3% | 9% | 9% |
| rootsift_flann_global | 49% | 40% | 0% | 6% | 11% | 3% |
| tiny_roma_track | 23% | 20% | 3% | 3% | 37% | 0% |
| superpoint_superglue_official_track | 37% | 17% | 3% | 3% | 23% | 0% |
| superpoint_superglue_track | 37% | 9% | 0% | 0% | 26% | 0% |
| brisk_bf_global | 14% | 23% | 0% | 0% | 3% | 3% |
| akaze_bf_track | 14% | 11% | 0% | 0% | 3% | 0% |
| efficient_loftr_track | 0% | 0% | 0% | 0% | 0% | 0% |
