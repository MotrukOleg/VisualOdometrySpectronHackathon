#!/usr/bin/env bash
# Learned part of the benchmark (GPU machine). Videos use every 2nd keyframe (video00xxs2) to fit the deadline.
set -u
B="python scripts/run_map_benchmark.py --no-viz"
M=configs/map/methods
ALL="crops syn_easy syn_hard video0061s2 video0062s2"
ABL="crops syn_hard video0062s2"
# already done on crops / synthetic before the switch to sub-sampled videos
$B --tag methods --configs $M/track_superpoint_lg.yaml $M/track_disk_lg.yaml --datasets video0061s2 video0062s2
for c in track_aliked_lg track_sift_lg track_aliked_rot_mnn track_dedode_mnn track_superglue track_loftr \
         track_eloftr track_xfeat track_xfeat_star track_tiny_roma; do
  $B --tag methods --configs $M/$c.yaml --datasets $ALL
done
$B --tag methods --configs $M/track_sift.yaml --datasets video0061s2 video0062s2
$B --tag methods --configs $M/track_roma.yaml --datasets crops syn_easy syn_hard video0062s2
$B --tag rotation --configs $M/track_superpoint_lg.yaml $M/track_loftr.yaml --datasets $ABL \
   --variants "tracking.rotation=none" "tracking.rotation=test"
$B --tag ablation_track_splg --configs $M/track_superpoint_lg.yaml --datasets $ABL --variants \
   "tracking.work_scale=2.0" "tracking.auto_scale=false" "tracking.motion_model=static" \
   "preprocess.blur_sigma=1.0" "preprocess.screen_mask=true" "preprocess.clahe=true" "geometry.estimator=ransac"
echo LEARNED_DONE
