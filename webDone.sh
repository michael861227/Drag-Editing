CUDA_VISIBLE_DEVICES=3 python drag_3d.py --config configs/main.yaml \
            --colmap_dir ./data/face/ \
            --gs_source ./data/face/point_cloud.ply \
            --point_dir ./logs/drag1/export/drag_points.json \
            --mask_dir ./logs/drag1/export/gaussian_mask.pt  \
            --output_dir drag1-iter2500