import os
import torch
import random
import numpy as np
import imageio
import tqdm
from utils import sample_from_dense_cameras, export_ply_for_gaussians, import_str, matrix_to_square, export_video
from PIL import Image
import torch.nn.functional as F
from modules.optimizers.gs_optimizer import GSOptimizer
from modules.optimizers.gs_utils import load_ply
from modules.scene.camera_scene import CamScene

from torchvision.utils import save_image
import shutil
from utils import seed_everything
from rich.console import Console
import cv2

import argparse
from omegaconf import OmegaConf
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    #output dir
    parser.add_argument("--base_dir", default='logs/')
    parser.add_argument("--output_dir", default='./exps/tmp')
    #input
    parser.add_argument("--gs_source", default=None)
    parser.add_argument("--colmap_dir", default=None)
    parser.add_argument("--point_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    
    parser.add_argument("--gpu", type=int, default=0)
    
    parser.add_argument("--from_director3d", action='store_true', default=False)
    
    # 新增多階段參數
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages to divide the drag operation")
    parser.add_argument("--lora_only_first_stage", action='store_true', default=False, help="Only train LoRA in first stage, reuse weights in subsequent stages")

    args, extras = parser.parse_known_args()
    args.out_dir = os.path.join(args.base_dir, args.output_dir)
    
    print(args)

    opt = OmegaConf.load(args.config)
    CONSOLE = Console()
    #if input from the command line
    if args.gs_source is not None:
        opt['scene']['gs_source'] = args.gs_source
    if args.colmap_dir is not None:
        opt['scene']['colmap_dir'] = args.colmap_dir
    if args.point_dir is not None:
        with open(args.point_dir, "r") as file:
            data = json.load(file)
        opt['scene']['handle_points'] = data.get("handle_points", [])
        opt['scene']['target_points'] = data.get("target_points", [])
    if args.mask_dir is not None:
        opt['scene']['mask_dir'] = args.mask_dir

    torch.backends.cudnn.benchmark = True

    device = f'cuda:{args.gpu}'
    
    # move config files to out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.out_dir, 'config.yaml'))

    seed_everything(0)

    if args.from_director3d:
        camera_npy_files = [f for f in os.listdir(opt['scene']['colmap_dir']) if f.endswith('.npy')]
        assert len(camera_npy_files) == 1
        #B N K
        camera_npy = np.load(os.path.join(opt['scene']['colmap_dir'], camera_npy_files[0]))
        camera_npy = torch.tensor(camera_npy,device=device)[0]
        cameras_extent = 2
        colmap_cameras = []
        for i in range(camera_npy.shape[0]):
            from utils import convert_camera_parameters_into_viewpoint_cameras
            colmap_cameras.append(convert_camera_parameters_into_viewpoint_cameras(camera_npy[i],image_name=f'frame_{i:05}',render_res_factor=opt['scene']['render_res_factor']))
    else:
        scene = CamScene(opt['scene']['colmap_dir'], h=-1, w=-1, render_res_factor=opt['scene']['render_res_factor'])
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.cameras

    # 載入原始edit_mask和gaussian
    original_edit_mask = torch.load(opt['scene']['mask_dir']).cuda()
    original_gaussians = load_ply(opt['scene']['gs_source'])
    
    handle_points = opt['scene']['handle_points']
    target_points = opt['scene']['target_points']

    # 初始化階段性拖曳的起始點和最終目標點
    original_handle_points = handle_points
    final_target_points = target_points
    original_handle_points_tensor = torch.tensor(original_handle_points, device=device)
    final_target_points_tensor = torch.tensor(final_target_points, device=device)
    
    # 多階段拖曳
    for stage in range(1, args.num_stages + 1):
        print(f"\n=== Starting Stage {stage}/{args.num_stages} ===")
        
        # 設定當前階段的輸出目錄
        stage_output_dir = os.path.join(args.out_dir, f'stage{stage}')
        os.makedirs(stage_output_dir, exist_ok=True)
        
        # 載入當前階段的gaussian和計算當前階段的handle points
        if stage == 1:
            # 第一階段使用原始gaussian和原始handle points
            current_gaussians = original_gaussians
            current_edit_mask = original_edit_mask
            current_handle_points = original_handle_points
        else:
            # 後續階段載入上一階段的結果
            prev_stage_dir = os.path.join(args.out_dir, f'stage{stage-1}')
            current_gaussians = load_ply(os.path.join(prev_stage_dir, 'result.ply'))
            
            # 根據上一階段的masks_lens_group重建edit_mask
            masks_info_path = os.path.join(prev_stage_dir, 'masks_info.json')
            if os.path.exists(masks_info_path):
                with open(masks_info_path, 'r') as f:
                    masks_info = json.load(f)
                masks_lens_group = masks_info['masks_lens_group']
                
                # 重建edit_mask：前masks_lens_group[0]個點是被編輯的點
                xyz, _, _, _, _ = current_gaussians
                current_edit_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=device)
                current_edit_mask[:masks_lens_group[0]] = True
                
                # 當前階段的handle points是上一階段的target points
                current_handle_points = masks_info['target_points']
            else:
                # 如果沒有masks_info，使用原始mask（這種情況不應該發生）
                CONSOLE.log(f"Warning: masks_info.json not found for stage {stage-1}, using original mask", style="red")
                current_edit_mask = original_edit_mask
                current_handle_points = original_handle_points
        
        # 計算當前階段的目標點
        stage_ratio = stage / args.num_stages
        current_target_points_tensor = original_handle_points_tensor + stage_ratio * (final_target_points_tensor - original_handle_points_tensor)
        current_target_points = current_target_points_tensor.tolist()
        
        CONSOLE.log(f"Stage {stage} - Handle: {current_handle_points} | Origin: {original_handle_points}", style="yellow")
        CONSOLE.log(f"Stage {stage} - Target: {current_target_points} | Final: {final_target_points}", style="cyan")
        # 創建GSOptimizer
        # Skip LoRA training if lora_only_first_stage is True and this is not the first stage
        skip_lora_training = args.lora_only_first_stage and stage > 1
        gsoptimizer = GSOptimizer(**opt['gsoptimizer']['args'],
                                  image_height=colmap_cameras[0].image_height//colmap_cameras[0].render_res_factor,
                                  image_width=colmap_cameras[0].image_width//colmap_cameras[0].render_res_factor,
                                  train_args=args,
                                  skip_lora_training=skip_lora_training,
                                  ).to(device)
        
        # 設定輸出目錄到當前階段
        gsoptimizer.output_dir = stage_output_dir
        
        # 準備gaussian mask
        xyz, features, opacity, scales, rotations = current_gaussians
        gaussians_mask = (xyz[current_edit_mask],features[current_edit_mask],opacity[current_edit_mask],scales[current_edit_mask],rotations[current_edit_mask])

        #delete invaild camera pose
        train_colmap_cameras = []
        for i,cam in enumerate(colmap_cameras):
            rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam],current_gaussians, scaling_modifier=1.0, bg_color=None)

            assert len(current_handle_points) == len(current_target_points)
            vaild_flag = True
            for j in range(len(current_handle_points)):
                h_pixel,h_depth = cam.world_to_pixel(current_handle_points[j])
                g_pixel,g_depth = cam.world_to_pixel(current_target_points[j])
                if h_depth is None or g_depth is None: 
                    vaild_flag = False
            
            rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam],current_gaussians, scaling_modifier=1.0, bg_color=None)
            rendered_image_mask, rendered_depth_mask, _ = gsoptimizer.renderer([cam],gaussians_mask, scaling_modifier=1.0, bg_color=None)
            mask_image = torch.ones_like(rendered_depth_mask)
            mask_image[rendered_depth_mask==cam.zfar] = 0
            mask_image[rendered_depth<rendered_depth_mask - opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
            if (mask_image==1).sum() < 50:
                vaild_flag = False

            if vaild_flag:
                train_colmap_cameras.append(cam)

        #valid camera number must > 10
        assert len(train_colmap_cameras) > 10

        #output init image and point for current stage
        for i,cam in enumerate(train_colmap_cameras):
            rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam],current_gaussians, scaling_modifier=1.0, bg_color=None)
            
            import cv2
            img = (rendered_image/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            assert len(current_handle_points) == len(current_target_points)
            for j in range(len(current_handle_points)):
                h_pixel,_ = cam.world_to_pixel(current_handle_points[j])
                g_pixel,_ = cam.world_to_pixel(current_target_points[j])
                cv2.circle(img, (int(h_pixel[0]),int(h_pixel[1])), 8, (255, 0, 0), -1)
                cv2.circle(img, (int(g_pixel[0]),int(g_pixel[1])), 8, (0, 0, 255), -1)
                cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
                                     (int(g_pixel[0]),int(g_pixel[1])), (255, 255, 255), 2, tipLength=0.5)
            init_dir = os.path.join(stage_output_dir,"init")
            if not os.path.exists(init_dir):
                os.makedirs(init_dir)
            cv2.imwrite(f"{init_dir}/cam{i+1}_{cam.image_name}.png",img)

            rendered_image_mask, rendered_depth_mask, _ = gsoptimizer.renderer([cam],gaussians_mask, scaling_modifier=1.0, bg_color=None)

            mask_image = torch.ones_like(rendered_depth_mask)
            mask_image[rendered_depth_mask==cam.zfar] = 0
            mask_image[rendered_depth<rendered_depth_mask - opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
            mask_image = mask_image.squeeze(0).permute(1, 2, 0).repeat(1,1,3).detach().cpu().numpy()
            mask_image = (mask_image.clip(min = 0, max = 1)*255.0).astype(np.uint8)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)

            mask_dilate_size, mask_dilate_iter = opt['gsoptimizer']['args']['mask_dilate_size'], opt['gsoptimizer']['args']['mask_dilate_iter']
            mask_image_dilate = cv2.dilate(mask_image,np.ones((mask_dilate_size,mask_dilate_size),np.int8),iterations=mask_dilate_iter)

            image_with_mask = ((0.3 * mask_image_dilate + img).clip(min = 0, max = 255) - mask_image).clip(min = 0, max = 255).astype(np.uint8)
            input_info_dir = os.path.join(stage_output_dir,"input_info")
            if not os.path.exists(input_info_dir):
                os.makedirs(input_info_dir)
            init_img_and_mask= cv2.hconcat([img, mask_image, mask_image_dilate, image_with_mask])
            cv2.imwrite(f"{input_info_dir}/cam{i+1}_init_img_and_mask.png", init_img_and_mask)

        # 如果需要在後續階段使用第一階段的LoRA權重，先進行特殊準備
        if args.lora_only_first_stage and stage > 1:
            # 為後續階段載入第一階段的LoRA權重
            lora_weights_path = os.path.join(args.out_dir, 'lora_weights')
            gsoptimizer.prepare_embeddings_with_lora_control(
                current_gaussians, train_colmap_cameras, 
                current_handle_points, current_target_points,
                gsoptimizer.image_height, gsoptimizer.image_width,
                lora_weights_path=lora_weights_path
            )
        
        # 執行當前階段的拖曳訓練
        result_gaussians, masks_lens_group, training_params = gsoptimizer.train_drag(current_gaussians,train_colmap_cameras,cameras_extent,current_edit_mask,current_handle_points,current_target_points)
        
        # 如果這是第一階段且啟用了lora_only_first_stage，保存LoRA權重並執行推理
        if args.lora_only_first_stage and stage == 1:
            lora_weights_path = os.path.join(args.out_dir, 'lora_weights')
            gsoptimizer.save_lora_weights(lora_weights_path)
            CONSOLE.log(f"LoRA weights saved to {lora_weights_path} after stage 1", style="green")
            
            # 執行LoRA推理，為每個訓練視角生成一張輸出圖片
            lora_denoise_dir = os.path.join(args.out_dir, 'lora_denoise')
            os.makedirs(lora_denoise_dir, exist_ok=True)
            CONSOLE.log(f"Starting LoRA inference for {len(train_colmap_cameras)} training views...", style="blue")
            
            # 為每個訓練視角生成推理圖片
            for i, cam in enumerate(train_colmap_cameras):
                # 渲染當前視角的圖像
                rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam], current_gaussians, scaling_modifier=1.0, bg_color=None)
                
                # 準備mask圖像
                rendered_image_mask, rendered_depth_mask, _ = gsoptimizer.renderer([cam], gaussians_mask, scaling_modifier=1.0, bg_color=None)
                mask_image = torch.ones_like(rendered_depth_mask)
                mask_image[rendered_depth_mask==cam.zfar] = 0
                mask_image[rendered_depth<rendered_depth_mask - opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
                
                # 創建dilated mask
                mask_dilate_size, mask_dilate_iter = opt['gsoptimizer']['args']['mask_dilate_size'], opt['gsoptimizer']['args']['mask_dilate_iter']
                mask_np = mask_image.squeeze(0).permute(1, 2, 0).repeat(1,1,3).detach().cpu().numpy()
                mask_np = (mask_np.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2BGR)
                mask_image_dilate = cv2.dilate(mask_np, np.ones((mask_dilate_size, mask_dilate_size), np.int8), iterations=mask_dilate_iter)
                
                # 轉換為PIL圖像格式
                mask_pil = Image.fromarray(cv2.cvtColor(mask_image_dilate, cv2.COLOR_BGR2RGB))
                
                # 準備handle和target points的像素坐標
                handle_points_pixel = []
                target_points_pixel = []
                for j in range(len(current_handle_points)):
                    h_pixel, _ = cam.world_to_pixel(current_handle_points[j])
                    t_pixel, _ = cam.world_to_pixel(current_target_points[j])
                    handle_points_pixel.append([int(h_pixel[0]), int(h_pixel[1])])
                    target_points_pixel.append([int(t_pixel[0]), int(t_pixel[1])])
                
                # 使用LoRA進行推理 - 確保完整的dtype一致性
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    # 編碼渲染圖像到latent空間
                    rendered_image_norm = rendered_image * 0.5 + 0.5  # Convert from [-1,1] to [0,1]
                    latents = gsoptimizer.pipe.vae.encode(rendered_image_norm.to(dtype=gsoptimizer.pipe.vae.dtype)).latent_dist.sample()
                    latents = latents * gsoptimizer.pipe.vae.config.scaling_factor
                    
                    # 使用與訓練最終iteration相同的時間步進行推理
                    t = torch.tensor([training_params['final_t']], dtype=torch.long, device=gsoptimizer.device)
                    
                    # 添加噪聲
                    noise = torch.randn_like(latents)
                    latents_noisy = gsoptimizer.pipe.scheduler.add_noise(latents, noise, t)
                    
                    # 確保所有輸入都是float32並且整個unet_lora模型也是float32
                    latents_noisy = latents_noisy.float()
                    embeddings = gsoptimizer.pipe.learnable_embeddings.repeat(1, 1, 1).float()
                    
                    # 確保unet_lora模型所有參數都是float32
                    original_unet_dtype = next(gsoptimizer.pipe.unet_lora.parameters()).dtype
                    if original_unet_dtype != torch.float32:
                        gsoptimizer.pipe.unet_lora = gsoptimizer.pipe.unet_lora.float()
                    
                    # 使用LoRA模型進行去噪
                    noise_pred = gsoptimizer.pipe.unet_lora(
                        latents_noisy,
                        t,
                        encoder_hidden_states=embeddings,
                    ).sample
                    
                    # 恢復原始dtype（如果有改變的話）
                    if original_unet_dtype != torch.float32:
                        gsoptimizer.pipe.unet_lora = gsoptimizer.pipe.unet_lora.to(original_unet_dtype)
                    
                    # 計算去噪後的latents
                    alpha_prod_t = gsoptimizer.pipe.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (latents_noisy - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                    
                    # 解碼回圖像空間
                    result_image = gsoptimizer.pipe.vae.decode(pred_original_sample.to(dtype=gsoptimizer.pipe.vae.dtype)).sample
                
                # 轉換到[0,1]範圍並保存
                result_image = (result_image * 0.5 + 0.5).clamp(0, 1)
                result_image_pil = gsoptimizer.pipe.image_processor.postprocess(result_image, output_type='pil')[0]
                
                # 保存推理結果
                output_path = os.path.join(lora_denoise_dir, f'cam_{i+1}_{cam.image_name}_lora_denoise.png')
                result_image_pil.save(output_path)
                        
             
            CONSOLE.log(f"LoRA inference completed. Results saved to {lora_denoise_dir}", style="green")
        
        #* Generate guidance images after training completes
        if stage_output_dir is not None:
            torch.cuda.empty_cache()  # Clear training memory first
            
            # Prepare gaussians_init and gs_init_mask for guidance generation
            xyz, features, opacity, scales, rotations = current_gaussians
            gaussians_init = current_gaussians
            gs_init_mask = (xyz[current_edit_mask],features[current_edit_mask],opacity[current_edit_mask],scales[current_edit_mask],rotations[current_edit_mask])
            
            # Pass the exact training parameters to ensure consistency
            gsoptimizer.generate_final_guidance_images(
                result_gaussians, gaussians_init, gs_init_mask, 
                train_colmap_cameras, current_handle_points, current_target_points, stage_output_dir, training_params
            )
            
        
        # 保存當前階段的masks_lens_group信息
        # 將tensor轉換為python基本類型以便JSON序列化
        masks_lens_group_serializable = []
        for item in masks_lens_group:
            if isinstance(item, torch.Tensor):
                masks_lens_group_serializable.append(item.item())
            else:
                masks_lens_group_serializable.append(int(item))
        
        # 將OmegaConf ListConfig轉換為普通Python list
        def convert_to_serializable(obj):
            """將OmegaConf或其他特殊對象轉換為JSON可序列化的格式"""
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):  # list-like objects
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # tensor-like objects
                return obj.item()
            else:
                return obj
        
        masks_info = {
            'masks_lens_group': masks_lens_group_serializable,
            'stage': stage,
            'handle_points': convert_to_serializable(current_handle_points),
            'target_points': convert_to_serializable(current_target_points)
        }
        with open(os.path.join(stage_output_dir, 'masks_info.json'), 'w') as f:
            json.dump(masks_info, f, indent=2)
        
        print(f"Stage {stage} completed. Results saved to {stage_output_dir}")
        print(f"  - Final masks for next stage saved to: {stage_output_dir}/mask/")
        print(f"    Files follow format: cam_{{i+1}}_{{frame_name}}.png")

    print(f"\n=== All {args.num_stages} stages completed ===")

    