CUDA_VISIBLE_DEVICES=0 python drag_3d.py --config configs/main.yaml \
            --colmap_dir ./data/face/ \
            --gs_source ./data/face/point_cloud.ply \
            --point_dir ./logs/drag1/export/drag_points.json \
            --mask_dir ./data/face/export_1/gaussian_mask.pt  \
            --output_dir drag1-export1-3Stage-test1 \
            --num_stages 3 \
            --lora_only_first_stage