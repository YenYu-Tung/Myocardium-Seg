# Computed Tomography Myocardium Image Segmentation

## 專案結構

```
project_root/
│
├─ workspace/
│  ├─ augmentation/
│  ├─ checkpoints/
│  ├─ core/
│  ├─ data_utils/
│  ├─ dataset/
│  │  └─ chgh/                # patient0001.nii.gz / patient0001_gt.nii.gz
│  ├─ entrypoints/
│  ├─ networks/
│  └─ optimizers/
│
├─ aicup_training.py
├─ aicup_inference.py
├─ ensemble_masks.py
│
├─ 41_training_image_*/
├─ 41_training_label/
├─ 41_testing_image_*/
│
├─ requirements.txt
└─ README.md
```
## 資料準備
- 原始資料解壓縮成以下資料夾：41_training_image_*/、41_training_label/、41_testing_image_*/，這些資料夾需放在專案根目錄
- 用於訓練的影像與標註需再放入 `workspace/dataset/chgh/`，檔名對應 `patientXXXX.nii.gz` 與 `patientXXXX_gt.nii.gz`。
- 預設讀取 `workspace/dataset/`；若路徑不同，用 `--train-image-dirs`、`--label-dir` 指定。
- K-fold：未提供 `--data_dicts_json` 時，可透過 `--num-folds` / `--fold-index` 直接從 `dataset/` 生成。若已有 JSON（建議放在 `workspace/exps/data_dicts/...`），用 `--data_dicts_json` 指定。
- 只對測試資料夾（例如 `41_testing_image_*`）推論時，可用 `--img-pth` 指定路徑；或先生成 data_dicts JSON，再用 `--data-dicts-json`。

## 環境安裝

```bash
conda create -n ctseg python=3.10
conda activate ctseg
pip install -r requirements.txt
```

### GPU（依 CUDA 版本安裝對應 PyTorch）

CUDA 12.6 範例：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
其他版本請至 https://pytorch.org/ 取得對應指令。

## 訓練指令

- 強化增強 + AMP + EMA  
```bash
python aicup_training.py --model-name swinunetr --strong-aug --use-ema --ema-decay 0.999 --use-amp
```

- 加入類別權重  
```bash
python aicup_training.py --model-name swinunetr --class-weights 1 1 3 2
```

- 使用 AdaBelief  
```bash
python aicup_training.py --optim AdaBelief
```

- K-fold 交叉驗證  
```bash
python aicup_training.py --num-folds 5 --fold-index 0
```

- 預設訓練指令
```bash
python aicup_training.py --model-name swinunetr --strong-aug --use-ema --use-amp  --warmup-epochs 80 --warmup-start-lr 1e-5 --eta-min 1e-6 
```

## 推論

- 產生影像標註檔  
```bash
python aicup_inference.py --exp-name AICUP_training_local --zip-output file_name.zip --infer-post-process --checkpoint workspace/checkpoints/best_model.pth --model-name swinunetr
```

- 只計算指標（不輸出影像）  
```bash
python aicup_inference.py --exp-name AICUP_training_local --data-dicts-json workspace/exps/data_dicts/chgh/AICUP_training_local.json --checkpoint workspace/checkpoints/best_model.pth --model-name swinunetr --metrics-only
```

## 集成 (Ensemble)

```bash
python ensemble_masks.py --input-dirs dir_name_1 dir_name_2 dir_name_3 --output-dir ouput_dir_name --num-classes 4
```

## 驗證曲線繪製

```bash
python plot_val_curve.py --run-dirs tune_results_dir_name
```

## 避免錯誤警告
請避免將專案放在過多層級或資料夾名稱過長的路徑中，以免因訓練 log 檔案路徑過長而導致訓練過程出錯。
