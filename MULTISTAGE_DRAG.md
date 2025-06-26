# 多階段拖曳編輯功能 (Multi-stage Drag Editing)

## 功能概述

此功能允許將原本的一次性拖曳編輯分成多個階段來完成，提供更細緻的控制和更穩定的編輯結果。

## 使用方法

### 基本命令

```bash
python drag_3d.py --config configs/main.yaml \
                  --gs_source path/to/gaussians.ply \
                  --colmap_dir path/to/colmap \
                  --point_dir path/to/points.json \
                  --mask_dir path/to/mask.pt \
                  --num_stages 3 \
                  --output_dir experiments/multi_stage_test
```

### 新增參數

- `--num_stages`: 指定拖曳分成多少個階段 (預設為 1，即原本的單階段模式)

## 工作原理

### 階段性目標計算

假設從點 A 拖曳到點 B，分成 n 個階段：

- Stage 1: 目標點 = A + (B-A)/n
- Stage 2: 目標點 = A + 2\*(B-A)/n
- ...
- Stage n: 目標點 = B

### 階段性儲存結構

```
output_dir/
├── stage1/
│   ├── init/           # 初始圖像
│   ├── input_info/     # 輸入信息圖像
│   ├── optim_*/        # 訓練過程圖像
│   ├── result.ply      # 第一階段結果
│   ├── masks_info.json # mask追蹤信息
│   └── compare.mp4     # 比較視頻
├── stage2/
│   ├── ...
│   ├── result.ply      # 第二階段結果
│   └── masks_info.json
└── stage3/
    ├── ...
    ├── result.ply      # 最終結果
    └── masks_info.json
```

### Edit Mask 追蹤機制

使用現有的`masks_lens_group`機制來追蹤被編輯的 gaussian 點：

- 每個階段結束後保存`masks_lens_group`信息到`masks_info.json`
- 下一階段開始時，根據`masks_lens_group[0]`重建 edit_mask
- 確保在所有階段中編輯同一組邏輯上的 gaussian 點

## 範例使用場景

### 分 3 階段進行細緻拖曳

```bash
python drag_3d.py --config configs/main.yaml \
                  --gs_source data/scene.ply \
                  --colmap_dir data/colmap \
                  --point_dir data/drag_points.json \
                  --mask_dir data/edit_mask.pt \
                  --num_stages 3 \
                  --output_dir logs/gradual_drag
```

### 單階段模式（向下兼容）

```bash
python drag_3d.py --config configs/main.yaml \
                  --gs_source data/scene.ply \
                  --colmap_dir data/colmap \
                  --point_dir data/drag_points.json \
                  --mask_dir data/edit_mask.pt \
                  --output_dir logs/single_stage
```

## 技術細節

### Gaussian 追蹤

- 在訓練過程中，gaussian 可能會通過`densify_and_prune`操作增減
- 使用`GaussiansManager`的分組機制來追蹤不同類型的 gaussian：
  - `3dmask`: 原始被編輯的 gaussian 點
  - `knn`: KNN 相鄰點
  - `other`: 其他 gaussian 點

### 階段間數據傳遞

- 每階段的結果 gaussian 自動保存為下一階段的輸入
- `masks_info.json`記錄關鍵的追蹤信息
- 自動處理階段間的目錄結構和文件管理

## 優勢

1. **穩定性**: 逐步調整比一次性大幅度調整更穩定
2. **可控性**: 可以檢查每個階段的中間結果
3. **靈活性**: 可以根據需求調整階段數量
4. **兼容性**: 完全向下兼容原有的單階段模式

## 注意事項

- 建議從較少的階段數開始嘗試（2-3 個階段）
- 階段數過多可能會增加總訓練時間
- 每個階段都會生成完整的輸出文件，注意磁盤空間使用
