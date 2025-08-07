CUDA_VISIBLE_DEVICES=3 python drag_3d.py --config configs/main.yaml \
            --colmap_dir ./data/face/ \
            --gs_source ./data/face/point_cloud.ply \
            --point_dir ./logs/drag1/export/drag_points.json \
            --mask_dir ./data/face/export_1/gaussian_mask.pt  \
            --output_dir drag1-export1-mask \
            --num_stages 1