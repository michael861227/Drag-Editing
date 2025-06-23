CUDA_VISIBLE_DEVICES=1 python drag_3d.py --config configs/main.yaml \
            --colmap_dir ./data/face/ \
            --gs_source ./data/face/point_cloud.ply \
            --point_dir ./data/face/export_1/drag_points.json \
            --mask_dir ./data/face/export_1/gaussian_mask.pt  \
            --output_dir face-export1-2DTrain