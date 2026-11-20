# Any-area benchmark

Map: Esri World Imagery at zoom 17 (~0.8 m/px, 2.8×1.4 km); UAV frames rendered from zoom 19 (zoom 18 ×2 where 19 is missing: carpathians, poltava and the 5 hard areas), frame→map scale 0.25, exact GT. 80 frames (first four areas) or 50 frames (hard areas). success@5px = share of ALL frames within 5 map px (≈4 m) of the GT, unregistered frames count as failures.


## success@5px

| method | Lviv city | Kyiv city+river | Carpathians forest+resort | Oleshky sand dunes | Gorgany forest | Askania steppe | Tylihul estuary | Kyiv sea water | Poltava fields |
|---|---|---|---|---|---|---|---|---|---|
| loftr_track | 100% | 100% | 100% | 100% | 98% | 66% | 84% | 98% | 96% |
| disk_lightglue_track | 100% | 99% | 100% | 100% | 92% | 86% | 80% | 86% | 98% |
| sift_lightglue_track | 100% | 89% | 100% | 100% | 90% | 86% | 66% | 72% | 84% |
| orb_bf_track | 100% | 95% | 100% | 100% | 92% | 86% | 74% | 62% | 38% |
| efficient_loftr_official_track | 100% | 96% | 100% | 100% | 84% | 64% | 70% | 76% | 40% |
| aliked_lightglue_track | 100% | 88% | 100% | 100% | 90% | 84% | 64% | 62% | 35% |
| xfeat_track | 99% | 85% | 94% | 100% | 74% | 64% | 54% | 72% | 38% |
| superpoint_superglue_official_track | 100% | 71% | 100% | 100% | 72% | 62% | 78% | 62% | 30% |
| superpoint_lightglue_track | 100% | 71% | 100% | 100% | 86% | 62% | 60% | 62% | 32% |
| xfeat_star_track | 100% | 86% | 100% | 100% | 74% | 62% | 54% | 62% | 35% |
| superpoint_superglue_track | 100% | 71% | 100% | 100% | 78% | 62% | 64% | 62% | 29% |
| sift_mnn_global | 98% | 72% | 100% | 100% | 72% | 66% | 52% | 64% | 30% |
| aliked_n16rot_mnn_track | 100% | 71% | 100% | 100% | 76% | 62% | 52% | 62% | 29% |
| rootsift_flann_global | 100% | 71% | 100% | 100% | 72% | 62% | 52% | 62% | 30% |
| sift_flann_track | 100% | 71% | 100% | 100% | 72% | 62% | 52% | 62% | 29% |
| sift_adalam_track | 100% | 71% | 100% | 100% | 70% | 62% | 52% | 62% | 30% |
| map_baseline_sift | 100% | 71% | 100% | 100% | 70% | 62% | 52% | 62% | 29% |
| akaze_bf_track | 100% | 71% | 100% | 100% | 70% | 62% | 52% | 62% | 29% |
| sift_flann_global | 100% | 71% | 100% | 100% | 70% | 62% | 52% | 62% | 29% |
| brisk_bf_global | 94% | 71% | 98% | 98% | 72% | 56% | 36% | 60% | 12% |
| orb_bf_global | 91% | 57% | 81% | 94% | 66% | 52% | 48% | 58% | 34% |
| tiny_roma_track | 84% | 60% | 82% | 78% | 60% | 50% | 40% | 50% | 31% |
| akaze_bf_global | 92% | 41% | 64% | 76% | 14% | 26% | 18% | 0% | 4% |


## registered (status map)

| method | Lviv city | Kyiv city+river | Carpathians forest+resort | Oleshky sand dunes | Gorgany forest | Askania steppe | Tylihul estuary | Kyiv sea water | Poltava fields |
|---|---|---|---|---|---|---|---|---|---|
| loftr_track | 100% | 100% | 100% | 100% | 96% | 66% | 84% | 98% | 96% |
| disk_lightglue_track | 100% | 98% | 100% | 100% | 94% | 86% | 80% | 86% | 98% |
| sift_lightglue_track | 100% | 89% | 100% | 100% | 90% | 86% | 66% | 72% | 85% |
| orb_bf_track | 100% | 94% | 100% | 100% | 92% | 86% | 74% | 64% | 38% |
| efficient_loftr_official_track | 100% | 95% | 100% | 100% | 82% | 64% | 70% | 76% | 39% |
| aliked_lightglue_track | 100% | 86% | 99% | 100% | 86% | 84% | 64% | 62% | 35% |
| xfeat_track | 100% | 86% | 100% | 100% | 76% | 64% | 56% | 70% | 36% |
| superpoint_superglue_official_track | 100% | 71% | 100% | 100% | 72% | 62% | 78% | 62% | 30% |
| superpoint_lightglue_track | 100% | 71% | 100% | 100% | 86% | 62% | 60% | 62% | 32% |
| xfeat_star_track | 100% | 82% | 100% | 100% | 72% | 62% | 54% | 62% | 35% |
| superpoint_superglue_track | 100% | 71% | 100% | 100% | 78% | 62% | 64% | 62% | 29% |
| sift_mnn_global | 96% | 72% | 99% | 100% | 72% | 66% | 52% | 64% | 30% |
| aliked_n16rot_mnn_track | 100% | 71% | 99% | 100% | 76% | 62% | 52% | 62% | 29% |
| rootsift_flann_global | 100% | 71% | 99% | 100% | 72% | 62% | 52% | 62% | 30% |
| sift_flann_track | 100% | 71% | 99% | 100% | 72% | 62% | 52% | 62% | 29% |
| sift_adalam_track | 100% | 71% | 99% | 100% | 70% | 62% | 52% | 62% | 30% |
| map_baseline_sift | 100% | 71% | 99% | 100% | 70% | 62% | 52% | 62% | 29% |
| akaze_bf_track | 100% | 71% | 99% | 100% | 70% | 62% | 52% | 62% | 29% |
| sift_flann_global | 100% | 71% | 99% | 100% | 70% | 62% | 52% | 62% | 29% |
| brisk_bf_global | 92% | 70% | 98% | 98% | 72% | 56% | 36% | 60% | 12% |
| orb_bf_global | 89% | 57% | 80% | 94% | 64% | 52% | 48% | 58% | 38% |
| tiny_roma_track | 100% | 76% | 99% | 100% | 74% | 64% | 54% | 62% | 62% |
| akaze_bf_global | 90% | 41% | 54% | 76% | 14% | 26% | 22% | 0% | 4% |


## median centre error, map px (registered + interpolated frames)

| method | Lviv city | Kyiv city+river | Carpathians forest+resort | Oleshky sand dunes | Gorgany forest | Askania steppe | Tylihul estuary | Kyiv sea water | Poltava fields |
|---|---|---|---|---|---|---|---|---|---|
| loftr_track | 0.55 | 0.17 | 0.16 | 0.18 | 0.23 | 0.18 | 0.32 | 0.31 | 0.24 |
| disk_lightglue_track | 0.47 | 0.12 | 0.07 | 0.12 | 0.19 | 0.07 | 0.17 | 0.21 | 0.14 |
| sift_lightglue_track | 0.46 | 0.19 | 0.08 | 0.11 | 0.16 | 0.13 | 0.22 | 0.23 | 0.19 |
| orb_bf_track | 0.45 | 0.20 | 0.18 | 0.18 | 0.19 | 0.21 | 0.29 | 0.35 | 464.72 |
| efficient_loftr_official_track | 0.56 | 0.23 | 0.23 | 0.18 | 0.28 | 0.20 | 0.37 | 0.40 | 368.65 |
| aliked_lightglue_track | 0.60 | 0.36 | 0.20 | 0.20 | 0.33 | 0.26 | 0.68 | 0.39 | 579.99 |
| xfeat_track | 0.60 | 0.69 | 0.46 | 0.58 | 0.62 | 0.44 | 3.22 | 0.38 | 455.72 |
| superpoint_superglue_official_track | 0.43 | 0.25 | 0.19 | 0.20 | 0.31 | 0.28 | 0.34 | 0.38 | 715.98 |
| superpoint_lightglue_track | 0.44 | 0.37 | 0.19 | 0.22 | 0.33 | 0.23 | 0.57 | 0.38 | 632.27 |
| xfeat_star_track | 1.57 | 1.23 | 1.24 | 1.34 | 1.58 | 1.19 | 2.21 | 0.37 | 579.77 |
| superpoint_superglue_track | 0.62 | 0.33 | 0.21 | 0.33 | 0.32 | 0.32 | 0.42 | 0.37 | 758.60 |
| sift_mnn_global | 0.34 | 0.38 | 0.36 | 0.36 | 0.36 | 0.36 | 0.57 | 0.37 | 742.63 |
| aliked_n16rot_mnn_track | 0.33 | 0.36 | 0.36 | 0.36 | 0.36 | 0.35 | 0.55 | 0.38 | 758.89 |
| rootsift_flann_global | 0.34 | 0.37 | 0.36 | 0.36 | 0.38 | 0.36 | 0.44 | 0.37 | 742.61 |
| sift_flann_track | 0.46 | 0.11 | 0.06 | 0.07 | 0.12 | 0.06 | 0.47 | 0.18 | 758.75 |
| sift_adalam_track | 0.46 | 0.08 | 0.05 | 0.07 | 0.13 | 0.05 | 0.59 | 0.14 | 742.54 |
| map_baseline_sift | 0.33 | 0.37 | 0.36 | 0.36 | 0.37 | 0.35 | 0.55 | 0.38 | 758.90 |
| akaze_bf_track | 0.43 | 0.16 | 0.06 | 0.11 | 0.25 | 0.10 | 0.54 | 0.37 | 758.75 |
| sift_flann_global | 0.33 | 0.37 | 0.36 | 0.36 | 0.37 | 0.36 | 0.53 | 0.38 | 758.89 |
| brisk_bf_global | 0.61 | 0.57 | 0.40 | 0.52 | 0.42 | 0.41 | 58.83 | 0.40 | 882.11 |
| orb_bf_global | 1.10 | 1.03 | 0.90 | 0.81 | 1.01 | 0.79 | 5.87 | 0.84 | 610.85 |
| tiny_roma_track | 1.27 | 2.55 | 1.11 | 1.08 | 1.62 | 1.34 | 17.14 | 3.15 | 465.39 |
| akaze_bf_global | 0.53 | 8.64 | 0.33 | 0.30 | 672.27 | 0.24 | 879.33 | – | 1226.63 |


## speed, frames per second (local GTX 1650 / shared RTX 5060 Ti — not comparable across machines)

| method | Lviv city | Kyiv city+river | Carpathians forest+resort | Oleshky sand dunes | Gorgany forest | Askania steppe | Tylihul estuary | Kyiv sea water | Poltava fields |
|---|---|---|---|---|---|---|---|---|---|
| loftr_track | 7.0 | 7.0 | 7.1 | 6.5 | 6.7 | 8.4 | 9.1 | 6.5 | 8.4 |
| disk_lightglue_track | 9.5 | 9.5 | 8.8 | 7.3 | 7.2 | 8.3 | 9.3 | 8.2 | 8.2 |
| sift_lightglue_track | 6.5 | 7.5 | 6.8 | 6.3 | 6.0 | 7.0 | 8.7 | 6.5 | 8.0 |
| orb_bf_track | 25.8 | 31.7 | 28.5 | 23.2 | 25.9 | 25.2 | 39.3 | 25.4 | 24.4 |
| efficient_loftr_official_track | 9.9 | 11.8 | 11.2 | 9.8 | 9.6 | 11.5 | 13.8 | 8.6 | 20.2 |
| aliked_lightglue_track | 8.0 | 11.8 | 10.7 | 9.5 | 9.1 | 10.9 | 13.7 | 10.7 | 2.1 |
| xfeat_track | 17.8 | 20.1 | 19.2 | 15.7 | 14.2 | 15.5 | 24.4 | 9.9 | 28.2 |
| superpoint_superglue_official_track | 8.5 | 11.0 | 9.8 | 7.8 | 9.2 | 9.9 | 12.5 | 10.1 | 21.6 |
| superpoint_lightglue_track | 7.7 | 11.3 | 10.9 | 9.6 | 9.7 | 11.8 | 9.5 | 5.4 | 12.7 |
| xfeat_star_track | 22.8 | 26.4 | 24.5 | 20.5 | 21.3 | 25.2 | 33.6 | 6.5 | 21.5 |
| superpoint_superglue_track | 2.6 | 5.5 | 3.8 | 3.7 | 4.4 | 5.5 | 7.8 | 13.7 | 12.4 |
| sift_mnn_global | 2.4 | 8.4 | 5.5 | 5.9 | 9.1 | 7.3 | 64.7 | 4.2 | 81.3 |
| aliked_n16rot_mnn_track | 12.9 | 18.2 | 14.5 | 13.3 | 15.1 | 17.0 | 32.7 | 13.9 | 47.6 |
| rootsift_flann_global | 16.9 | 35.3 | 26.5 | 27.6 | 29.4 | 28.0 | 59.1 | 19.9 | 81.4 |
| sift_flann_track | 13.8 | 19.8 | 13.6 | 15.5 | 15.2 | 17.5 | 19.0 | 17.1 | 26.2 |
| sift_adalam_track | 9.6 | 13.2 | 9.4 | 8.6 | 10.3 | 11.5 | 17.4 | 13.2 | 24.6 |
| map_baseline_sift | 12.9 | 35.4 | 30.8 | 26.1 | 29.2 | 27.7 | 64.6 | 19.4 | 81.1 |
| akaze_bf_track | 29.4 | 31.9 | 30.3 | 25.1 | 24.5 | 26.8 | 24.7 | 14.3 | 34.6 |
| sift_flann_global | 15.9 | 34.7 | 27.9 | 28.2 | 29.7 | 27.0 | 61.2 | 20.9 | 81.6 |
| brisk_bf_global | 14.0 | 30.9 | 76.2 | 65.3 | 77.2 | 86.9 | 270.9 | 48.8 | 714.6 |
| orb_bf_global | 4.3 | 6.0 | 8.1 | 6.2 | 7.2 | 7.2 | 22.6 | 7.6 | 62.0 |
| tiny_roma_track | 4.2 | 5.9 | 4.5 | 4.8 | 6.9 | 7.6 | 9.9 | 10.1 | 3.4 |
| akaze_bf_global | 29.2 | 72.4 | 68.2 | 65.0 | 83.2 | 77.8 | 88.3 | 76.6 | 98.0 |
