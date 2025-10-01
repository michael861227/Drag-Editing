## 🔧 Installation

- clone this repo:

```
git clone https://github.com/michael861227/Drag-Editing.git
cd Drag-Editing
git submodule update --init --recursive
```

- set up a new conda environment

```
conda env create --file environment.yaml
conda activate DYG
```

## 📚 Data Preparation

Refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for reconstruction. We recommend setting the spherical harmonic degree to 0 for the reconstruction process.
Alternatively, you can use the [data](https://drive.google.com/drive/folders/19Jv3crbF7xMu1ouNoCH-mEH87ClykpuY?usp=sharing) we have prepared. Taking the face scene as an example, the file structure is as follows:

```
└── data
    └── face
        ├── export_1
        │   ├── drag_points.json
        │   └── gaussian_mask.pt
        ├── image
        ├── sparse
        └── point_cloud.ply
```

Since we use LightningDrag as the diffusion prior, you need to download the model following the instructions in [lightningdrag](https://github.com/magic-research/LightningDrag/blob/main/INSTALLATION.md#2-download-pretrained-models) and organize it according to the following structure.

```
└── checkpoints
    ├── dreamshaper-8-inpainting
    ├── lcm-lora-sdv1-5
    │   └── pytorch_lora_weights.safetensors
    ├── sd-vae-ft-ema
    │   ├── config.json
    │   ├── diffusion_pytorch_model.bin
    │   └── diffusion_pytorch_model.safetensors
    ├── IP-Adapter/models
    │   ├── image_encoder
    │   └── ip-adapter_sd15.bin
    └── lightning-drag-sd15
        ├── appearance_encoder
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        ├── point_embedding
        │   └── point_embedding.pt
        └── lightning-drag-sd15-attn.bin
```

## 🚋 Training

Start the WebUI using the following command. For detailed usage instructions, refer to [WebUI](./assets/webui-guide/webui.md)

### Launch the WebUI

```shell
python webui.py --colmap_dir <path to colmap dir> --gs_source <path to 3DGS ply> --output_dir <save path>
```

### Example

```shell
python webui.py --colmap_dir ./data/face/ --gs_source ./data/face/point_cloud.ply --output_dir result
```

### Training without WebUI (Recommended)

If you already have `gaussian_mask.pt` and `drag_points.json` saved from WebUI, you can train without WebUI by using the following command.

`--num_stages`
Specifies how many stages the entire drag editing process is divided into. Instead of applying a single large drag operation, the process is split into multiple smaller stages to achieve more stable and better results.

`--lora_only_first_stage`
If enabled, LoRA training will only occur during the first stage. Subsequent stages will reuse the pretrained LoRA from stage one instead of retraining it.

```shell
python drag_3d.py --config configs/main.yaml \
            --colmap_dir ./data/face/ \
            --gs_source ./data/face/point_cloud.ply \
            --point_dir ./data/face/export_1/drag_points.json \
            --mask_dir ./data/face/export_1/gaussian_mask.pt  \
            --output_dir result \
            --num_stages 3 \
            --lora_only_first_stage
```

<!-- ## Citation

```
@article{qu2025drag,
  title={Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting},
  author={Qu, Yansong and Chen, Dian and Li, Xinyang and Li, Xiaofan and Zhang, Shengchuan and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2501.18672},
  year={2025}
}
``` -->

<!-- ## License

Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)

The code is released for academic research use only.

If you have any questions, please contact me via [quyans@stu.xmu.edu.cn](mailto:quyans@stu.xmu.edu.cn). -->

## Acknowledgement

The implementation is base on [Drag Your Gaussian](https://github.com/Quyans/Drag-Your-Gaussian)
