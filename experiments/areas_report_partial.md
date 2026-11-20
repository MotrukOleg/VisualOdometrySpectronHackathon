# Any-area benchmark (partial snapshot, 2026-11-20 16:00)

Runs collected so far (vast classical complete, vast learned 38/54, local learned in progress). Errors (status=error) excluded.


## success@5px (share of frames within 5 map px of GT), sorted by mean

| method | lviv | kyiv | carpathians | poltava | oleshky | gorgany | kyiv_sea | askania | tylihul | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| superpoint_lightglue_track | 100% | – | – | – | – | – | – | – | – | 100% |
| loftr_track | 100% | 100% | 100% | 96% | 100% | 98% | 98% | 66% | 84% | 94% |
| disk_lightglue_track | 100% | 99% | 100% | 98% | 100% | 92% | 86% | 86% | 80% | 93% |
| aliked_lightglue_track | 100% | 88% | 100% | – | 100% | 90% | 62% | 84% | – | 89% |
| orb_bf_track | 100% | 95% | 100% | – | 100% | 92% | 62% | 86% | 74% | 89% |
| sift_lightglue_track | 100% | 89% | 100% | 84% | 100% | 90% | 72% | 86% | 66% | 87% |
| efficient_loftr_official_track | 100% | 96% | 100% | – | 100% | 84% | 76% | 64% | 70% | 86% |
| tiny_roma_track | 84% | – | – | – | – | – | – | – | – | 84% |
| superpoint_superglue_official_track | 100% | 71% | 100% | – | 100% | 72% | – | 62% | 78% | 83% |
| superpoint_superglue_track | 100% | 71% | 100% | – | 100% | 78% | – | 62% | 64% | 82% |
| aliked_n16rot_mnn_track | 100% | 71% | 100% | – | 100% | 76% | 62% | 62% | – | 82% |
| sift_flann_track | 100% | 71% | 100% | – | 100% | 72% | 62% | 62% | – | 81% |
| akaze_bf_track | 100% | 71% | 100% | – | 100% | 70% | 62% | 62% | – | 81% |
| sift_adalam_track | 100% | 71% | 100% | – | 100% | 70% | 62% | 62% | – | 81% |
| sift_mnn_global | 98% | 72% | 100% | 30% | 100% | 72% | 64% | 66% | 52% | 73% |
| rootsift_flann_global | 100% | 71% | 100% | 30% | 100% | 72% | 62% | 62% | 52% | 72% |
| map_baseline_sift | 100% | 71% | 100% | 29% | 100% | 70% | 62% | 62% | 52% | 72% |
| sift_flann_global | 100% | 71% | 100% | 29% | 100% | 70% | 62% | 62% | 52% | 72% |
| brisk_bf_global | 94% | 71% | 98% | 12% | 98% | 72% | 60% | 56% | 36% | 66% |
| orb_bf_global | 91% | 57% | 81% | 34% | 94% | 66% | 58% | 52% | 48% | 65% |
| akaze_bf_global | 92% | 41% | 64% | 4% | 76% | 14% | 0% | 26% | 18% | 37% |


## registered (status map)

| method | lviv | kyiv | carpathians | poltava | oleshky | gorgany | kyiv_sea | askania | tylihul |
|---|---|---|---|---|---|---|---|---|---|
| superpoint_lightglue_track | 100% | – | – | – | – | – | – | – | – |
| loftr_track | 100% | 100% | 100% | 96% | 100% | 96% | 98% | 66% | 84% |
| disk_lightglue_track | 100% | 98% | 100% | 98% | 100% | 94% | 86% | 86% | 80% |
| aliked_lightglue_track | 100% | 86% | 99% | – | 100% | 86% | 62% | 84% | – |
| orb_bf_track | 100% | 94% | 100% | – | 100% | 92% | 64% | 86% | 74% |
| sift_lightglue_track | 100% | 89% | 100% | 85% | 100% | 90% | 72% | 86% | 66% |
| efficient_loftr_official_track | 100% | 95% | 100% | – | 100% | 82% | 76% | 64% | 70% |
| tiny_roma_track | 100% | – | – | – | – | – | – | – | – |
| superpoint_superglue_official_track | 100% | 71% | 100% | – | 100% | 72% | – | 62% | 78% |
| superpoint_superglue_track | 100% | 71% | 100% | – | 100% | 78% | – | 62% | 64% |
| aliked_n16rot_mnn_track | 100% | 71% | 99% | – | 100% | 76% | 62% | 62% | – |
| sift_flann_track | 100% | 71% | 99% | – | 100% | 72% | 62% | 62% | – |
| akaze_bf_track | 100% | 71% | 99% | – | 100% | 70% | 62% | 62% | – |
| sift_adalam_track | 100% | 71% | 99% | – | 100% | 70% | 62% | 62% | – |
| sift_mnn_global | 96% | 72% | 99% | 30% | 100% | 72% | 64% | 66% | 52% |
| rootsift_flann_global | 100% | 71% | 99% | 30% | 100% | 72% | 62% | 62% | 52% |
| map_baseline_sift | 100% | 71% | 99% | 29% | 100% | 70% | 62% | 62% | 52% |
| sift_flann_global | 100% | 71% | 99% | 29% | 100% | 70% | 62% | 62% | 52% |
| brisk_bf_global | 92% | 70% | 98% | 12% | 98% | 72% | 60% | 56% | 36% |
| orb_bf_global | 89% | 57% | 80% | 38% | 94% | 64% | 58% | 52% | 48% |
| akaze_bf_global | 90% | 41% | 54% | 4% | 76% | 14% | 0% | 26% | 22% |


## median centre error, map px

| method | lviv | kyiv | carpathians | poltava | oleshky | gorgany | kyiv_sea | askania | tylihul |
|---|---|---|---|---|---|---|---|---|---|
| superpoint_lightglue_track | 0.44 | – | – | – | – | – | – | – | – |
| loftr_track | 0.55 | 0.17 | 0.16 | 0.24 | 0.18 | 0.23 | 0.31 | 0.18 | 0.32 |
| disk_lightglue_track | 0.47 | 0.12 | 0.07 | 0.14 | 0.12 | 0.19 | 0.21 | 0.07 | 0.17 |
| aliked_lightglue_track | 0.60 | 0.36 | 0.20 | – | 0.20 | 0.33 | 0.39 | 0.26 | – |
| orb_bf_track | 0.45 | 0.20 | 0.18 | – | 0.18 | 0.19 | 0.35 | 0.21 | 0.29 |
| sift_lightglue_track | 0.46 | 0.19 | 0.08 | 0.19 | 0.11 | 0.16 | 0.23 | 0.13 | 0.22 |
| efficient_loftr_official_track | 0.56 | 0.23 | 0.23 | – | 0.18 | 0.28 | 0.40 | 0.20 | 0.37 |
| tiny_roma_track | 1.27 | – | – | – | – | – | – | – | – |
| superpoint_superglue_official_track | 0.43 | 0.25 | 0.19 | – | 0.20 | 0.31 | – | 0.28 | 0.34 |
| superpoint_superglue_track | 0.62 | 0.33 | 0.21 | – | 0.33 | 0.32 | – | 0.32 | 0.42 |
| aliked_n16rot_mnn_track | 0.33 | 0.36 | 0.36 | – | 0.36 | 0.36 | 0.38 | 0.35 | – |
| sift_flann_track | 0.46 | 0.11 | 0.06 | – | 0.07 | 0.12 | 0.18 | 0.06 | – |
| akaze_bf_track | 0.43 | 0.16 | 0.06 | – | 0.11 | 0.25 | 0.37 | 0.10 | – |
| sift_adalam_track | 0.46 | 0.08 | 0.05 | – | 0.07 | 0.13 | 0.14 | 0.05 | – |
| sift_mnn_global | 0.34 | 0.38 | 0.36 | 742.63 | 0.36 | 0.36 | 0.37 | 0.36 | 0.57 |
| rootsift_flann_global | 0.34 | 0.37 | 0.36 | 742.61 | 0.36 | 0.38 | 0.37 | 0.36 | 0.44 |
| map_baseline_sift | 0.33 | 0.37 | 0.36 | 758.90 | 0.36 | 0.37 | 0.38 | 0.35 | 0.55 |
| sift_flann_global | 0.33 | 0.37 | 0.36 | 758.89 | 0.36 | 0.37 | 0.38 | 0.36 | 0.53 |
| brisk_bf_global | 0.61 | 0.57 | 0.40 | 882.11 | 0.52 | 0.42 | 0.40 | 0.41 | 58.83 |
| orb_bf_global | 1.10 | 1.03 | 0.90 | 610.85 | 0.81 | 1.01 | 0.84 | 0.79 | 5.87 |
| akaze_bf_global | 0.53 | 8.64 | 0.33 | 1226.63 | 0.30 | 672.27 | – | 0.24 | 879.33 |


## failed runs (to rerun with the ROI fix)

                              label       dataset
             aliked_lightglue_track  area_poltava
             aliked_lightglue_track  area_tylihul
            aliked_n16rot_mnn_track  area_poltava
            aliked_n16rot_mnn_track  area_tylihul
                     akaze_bf_track  area_poltava
                   sift_flann_track  area_poltava
                       orb_bf_track  area_poltava
                     akaze_bf_track  area_tylihul
         superpoint_superglue_track  area_poltava
                   sift_flann_track  area_tylihul
                  sift_adalam_track  area_poltava
         superpoint_superglue_track area_kyiv_sea
                  sift_adalam_track  area_tylihul
     efficient_loftr_official_track  area_poltava
superpoint_superglue_official_track  area_poltava
superpoint_superglue_official_track area_kyiv_sea