CUDA_VISIBLE_DEVICES=2 python webui.py \
    --colmap_dir ./data/kitchen/ \
    --gs_source ./data/kitchen/point_cloud.ply \
    --output_dir kitchen-test