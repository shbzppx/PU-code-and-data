# PU-Learning Mineral Prospectivity Toolkit

This repository contains a complete PU-learning workflow for mineral prospectivity mapping: data preparation, model training, prediction map generation, metric evaluation, SHAP/DEC/P-A analysis, and model comparison/AutoML utilities.

Most user-facing tools are PyQt5 desktop applications. Several scripts can also be run from the command line for batch processing.

## Recommended Workflow

1. Prepare raw geoscience grids and mineral occurrence points.
2. Convert or fuse feature grids into HDF5 files.
3. Train PU-learning or baseline models.
4. Run model prediction to generate probability maps.
5. Evaluate predictions with GDR/SR/PAF/EI, SRC, P-A, DEC, SHAP, and related tools.
6. Optionally run model comparison or AutoML experiments.

## Environment

Use Python 3.9 or newer. A CUDA-enabled PyTorch installation is recommended for neural models, but most tools can run on CPU.

Install the common dependencies:

```bash
pip install numpy pandas h5py scipy scikit-learn matplotlib PyQt5 torch torchvision tqdm openpyxl pillow optuna shap
```

Depending on the exact workflow, optional packages such as `reportlab`, `seaborn`, or other plotting/report libraries may also be required.

On Windows PowerShell, run commands from the project root:


## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Main PyQt5 dashboard that launches the major workflow modules. |
| `数据准备/` | Data preparation GUI and standalone preprocessing scripts. |
| `feature/` | GRD-to-H5 feature fusion tools and patch creation utilities. |
| `模型训练/` | PU-learning training GUI, CLI training entry, dataset handling, losses, CV, metrics, visualization. |
| `模型预测/` | Prediction GUI, CLI prediction script, and probability map plotting tools. |
| `模型评估/` | Evaluation GUI and metric/interpretability tools including GDR, DEC, P-A, SHAP. |
| `model_comparison/` | Current model comparison and AutoML GUI for the system's built-in models. |
| `comparison/` | Older/general model comparison and AutoML module. |
| `model/` | Model implementations used by training, prediction, and comparison tools. |
| `cnn/`, `resnet/` | CNN/ResNet wrappers used by comparison modules. |
| `common/` | Shared helpers, especially feature-channel utilities. |
| `retro_spatial_cv_metrics.py` | CLI/GUI tool for recomputing spatial CV metrics for older training outputs. |
| `outputs/`, `模型训练/result/`, `模型预测/predictions/`, `模型评估/shap_outputs/` | Generated results and examples from previous runs. |

## Main Dashboard

Launch the integrated dashboard:

```bash
python main.py
```

The dashboard opens separate windows for:

- Data preparation
- Model training
- Model prediction
- Model evaluation
- Model comparison

If a module fails to open, run that module directly from the command line shown in the sections below. This makes dependency or path errors easier to diagnose.

## Data Preparation

### GUI Entry

```bash
python 数据准备/main.py
```

The data preparation GUI provides tabs for:

- Mineral point label interpolation against a reference grid.
- GRD file fusion into H5.
- DAT-to-H5 conversion.
- Multi-H5 feature merging.
- Sliding-window slicing.
- Spatial deduplication / spatially independent mineral point merging.

### Standalone Scripts

Run these scripts from inside `数据准备/` unless you pass absolute paths or edit defaults.

```bash
python 数据准备/2.interpolate_deposit.py
```

Interactive Tk file picker. Reads a mineral point `deposit.dat` and a reference grid such as `Ag.dat`, then writes a label DAT file with labels `1` for nearest mineral grid points and `-1` otherwise.

```bash
python 数据准备/3.convert_dat_to_h5.py
```

Interactive console script. Converts a 3-column DAT file (`X Y Value`) into H5 with datasets:

- `data`
- `x_coords`
- `y_coords`

```bash
python 数据准备/4.combine_h5_data.py
```

Interactive console script. Merges multiple H5 files into a combined H5 with:

- `coordinates`
- `vectors`

```bash
python 数据准备/5.export_combined_data.py -i combined_data.h5 -o combined_data.dat
```

Exports a combined H5 file back to DAT. If `-i` is omitted, the script lists H5 files in the current directory and asks you to choose one.

```bash
python 数据准备/6.slice_data.py
```

Interactive console script. Loads a combined H5 file, interpolates it onto a regular grid, applies padding, extracts sliding windows, and saves a windows H5 file with:

- `windows`
- `positions`
- `index_positions`
- coordinate metadata attributes

### GRD Feature Fusion GUI

```bash
python feature/feature_engineering_gui.py
```

Use this GUI to select one or more `.grd` files, order them, optionally apply normalization, and export a fused H5 feature file.

## Training

### GUI Entry

```bash
python 模型训练/main.py
```

Without arguments, the training script opens the PyQt5 GUI. It supports model selection, feature-channel selection, PU loss settings, spatial splitting, cross-validation, batch/grid experiments, and result visualization.

### CLI Entry

```powershell
python 模型训练/main.py ^
  --dataset path/to/features.h5 ^
  --label-path path/to/deposit_points.txt ^
  --model pucnntransformer ^
  --out 模型训练/result/run_example ^
  --device cuda ^
  --epoch 50 ^
  --batchsize 32 ^
  --patch-size 9 ^
  --patch-stride 1 ^
  --sample-ratio 0.1 ^
  --cv-folds 5
```

Use `--gui` with CLI defaults preloaded into the GUI:

```bash
python 模型训练/main.py --gui --dataset path/to/features.h5 --label-path path/to/deposit_points.txt
```

Important options:

| Option | Meaning |
| --- | --- |
| `--dataset` | Feature H5 file. |
| `--label-path` | Mineral coordinate TXT/CSV/DAT/H5 label file. |
| `--model` | `linear`, `3lp`, `mlp`, `cnn`, `pucnntransformer`, `rf`, `purf`, `ocsvm`, `2step`, etc. |
| `--loss-type` | `standard` or `adaptive`. |
| `--auto-prior` / `--manual-prior` | Prior probability handling. |
| `--split-mode` | `legacy`, `spatial_cluster`, `spatial_hard`, `spatial_stratified`, or `spatial_cluster_holdout_cv`. |
| `--selected-channels` | Zero-based comma-separated feature channel indices. Empty means all channels. |
| `--model-grid`, `--prior-grid`, `--patch-size-grid`, `--sample-ratio-grid`, `--gamma-grid` | Batch/grid experiment controls. |

Training outputs usually include:

- `model.pth` or `model.pkl`
- `model_fold*.pth` / `model_fold*.pkl` for CV runs
- `normalization_params.pth`
- `params.json`
- `cv_results.json`
- `metrics.json`
- training/loss/accuracy/recall curves
- spatial split reproducibility files
- `空间CV评估结果.xlsx`

### H5 Sampler

```bash
python 模型训练/random_h5_sampler.py
```

Samples matching feature and label H5 files while preserving positive/negative sample proportions. This script uses hard-coded defaults near the top:

- `INPUT_FILE1`
- `INPUT_FILE2`
- `SAMPLE_RATIO`

Edit these constants before running, or import `sample_h5_files()` from another script.

## Prediction

### GUI Entry

```bash
python 模型预测/main.py
```

The prediction GUI supports:

- Single or multiple model files.
- Neural `.pth` and sklearn-style `.pkl` models.
- H5 feature input.
- H5 or coordinate-based label/mineral files.
- CPU/CUDA/auto device selection.
- Probability map generation by center value, window average, or window max.
- CSV/DAT export.
- PAR threshold feedback.
- Loss-curve plotting from training output directories.

### CLI Prediction

```powershell
python 模型预测/predict.py ^
  --model-type pucnntransformer ^
  --model-path path/to/model.pth ^
  --input path/to/features.h5 ^
  --label-file path/to/deposit_points.txt ^
  --norm-params path/to/normalization_params.pth ^
  --prior 0.2 ^
  --batch-size 32 ^
  --img_size 9 ^
  --patch-stride 1 ^
  --output-dir 模型预测/predictions
```

For non-neural models, use the matching `.pkl` file and model type such as `rf`, `purf`, `ocsvm`, or `2step`.

Typical prediction output:

- `predictions_<model>.h5`
- probability/confidence datasets
- prediction labels
- positions
- optional mineral point metadata
- text metrics saved near the model directory

### Plot Prediction Maps

```powershell
python 模型预测/plot_labels.py ^
  --predictions 模型预测/predictions/predictions_pucnntransformer.h5 ^
  --model-name pucnntransformer ^
  --output-dir 模型预测/predictions/label_plots ^
  --map-generation-mode window_average
```

Supported map modes:

- `center`
- `window_average`
- `window_max`

The script exports DAT/CSV-compatible prediction tables and confidence map images.

## Evaluation

### GUI Entry

```bash
python 模型评估/main.py
```

The evaluation GUI supports:

- Loading prediction H5/DAT files.
- Direct evaluation from trained model files.
- GDR/SR/PAF/EI-style metrics.
- Independent test area evaluation.
- DEC and SRC curves.
- P-A curves.
- Confusion matrix display.
- SHAP-related model interpretation workflows.
- Exporting plots and CSV summaries.

### GDR Calculator

`模型评估/GDR_calculator.py` can be used as a library or script. By default, it uses hard-coded preset paths near the top of the file:

- `PREDICTION_FILE`
- `DEPOSIT_FILE`
- `DISTANCE_THRESHOLD`
- `CONFIDENCE_THRESHOLD`
- `USE_COMMAND_LINE`

To use command-line arguments, set `USE_COMMAND_LINE = True` in the script, then run:

```powershell
python 模型评估/GDR_calculator.py path/to/predictions.h5 ^
  --deposit_file path/to/deposit_points.csv ^
  --distance_threshold 5 ^
  --confidence_threshold 0.5
```

Prediction input can be H5 or DAT. H5 files should contain position and confidence/probability datasets such as `positions` and `confidences`.

### Other Evaluation Scripts

| Script | Purpose |
| --- | --- |
| `模型评估/evaluate_shap_cnnt.py` | SHAP analysis for CNN/Transformer-style trained models. |
| `模型评估/DEC.py` | DEC/SRC curve generation utilities. |
| `模型评估/P-A.py` | P-A curve calculation/plotting. |
| `模型评估/cross_validation.py` | Evaluation-side CV helpers. |
| `模型评估/predict.py` | Evaluation-side prediction helper. |
| `模型评估/evaluation有p.py` | Evaluation metric utilities with prior-probability related calculations. |

Some of these scripts are intended to be imported by the GUI rather than run directly. If a script contains hard-coded default paths, edit those paths before standalone execution.

## Model Comparison and AutoML

### Current System Model Comparison

```bash
python model_comparison/main.py
```

This is the preferred model-comparison GUI for the system's built-in models. It supports:

- Loading a full H5 feature dataset.
- Loading train/validation/test mineral files, single mineral files, manual mineral splitting, region-based splitting, or spatial splitting.
- Selecting feature channels/layer schemes.
- Comparing built-in models.
- AutoML workflow with stage 1/stage 2 trial settings.
- Batch repeated splits.
- Result tables, curves, artifact management, and PDF report export.

See also:

```text
model_comparison/使用指南-自动优化与模型管理.md
model_comparison/README.md
```

### Older / General Comparison Module

The `comparison/` directory contains an older/general comparison module for CNN, ResNet, Random Forest, SVM, and Decision Tree workflows.

There is no `comparison/main.py` in this copy. To test the placeholder GUI:

```bash
python comparison/test_gui.py
```

Most real functionality is implemented in `comparison/comparison_gui.py` and supporting modules. It is normally imported by a launcher or another GUI.

See also:

```text
comparison/使用指南-自动优化与模型管理.md
comparison/README.md
```

## Retroactive Spatial CV Metrics

Use this tool to recompute spatial CV SR/PAF/EI metrics for older training result directories.

Open GUI:

```bash
python retro_spatial_cv_metrics.py
```

Run from CLI:

```powershell
python retro_spatial_cv_metrics.py ^
  --model-dir path/to/old_training_result ^
  --device cpu ^
  --batch-size 256 ^
  --threshold-step 0.01 ^
  --distance-threshold 4 ^
  --output-json spatial_cv_retro_metrics.json ^
  --output-csv spatial_cv_retro_metrics.csv
```

Check metadata only:

```bash
python retro_spatial_cv_metrics.py --model-dir path/to/old_training_result --dry-run
```

The target training result directory must contain:

- `params.json`
- `normalization_params.pth`
- fold model files such as `model_fold1.pth`, ..., `model_fold5.pth`

## Core Library Modules

These directories are usually imported by GUIs or scripts rather than run directly:

| Path | Role |
| --- | --- |
| `model/` | Linear, MLP, CNN, CNN-Transformer, CNN-TokenTransformer, Random Forest, PU-RF, One-Class SVM, and Two-Step PU implementations. |
| `cnn/` | CNN model, data loader, and trainer used by comparison tools. |
| `resnet/` | ResNet implementations used by comparison tools. |
| `common/feature_channel_utils.py` | Shared feature-channel parsing, naming, and selection helpers. |
| `model_name_utils.py` | Shared model key/display-name normalization. |

## Input Data Notes

Common feature H5 formats:

1. Raw grid-style H5:
   - `coordinates`
   - `vectors`

2. Converted single-channel grid H5:
   - `data`
   - `x_coords`
   - `y_coords`

3. Pre-cut window H5:
   - `windows`
   - `positions`
   - optional `index_positions`
   - coordinate metadata attributes

Mineral point files can usually be TXT, DAT, CSV, TSV, XLS, or XLSX. Most tools try to detect X/Y columns from names such as:

- `x`, `coord_x`, `point_x`, `east`, `easting`, `longitude`
- `y`, `coord_y`, `point_y`, `north`, `northing`, `latitude`

If no recognized header exists, the first two columns are generally treated as X and Y.

## Output Management

Generated files can become large. Common output locations are:

- `模型训练/result/`
- `模型预测/predictions/`
- `模型预测/predictions/label_plots/`
- `模型评估/shap_outputs/`
- `outputs/`
- `model_comparison/outputs/` or `outputs/model_comparison/`

For reproducible runs, keep each experiment in a separate output directory and preserve `params.json`, `normalization_params.pth`, model files, and spatial split files together.

## Troubleshooting

- If a GUI does not open, confirm `PyQt5` is installed and run the specific module directly to see the error message.
- If CUDA is unavailable, choose `cpu` in the GUI or pass `--device cpu`.
- If prediction fails because of input shape mismatch, use the same `patch_size`, `patch_stride`, selected feature channels, and normalization file used during training.
- If coordinate-based metrics look wrong, verify that feature grids, prediction positions, and mineral point files use the same coordinate system and units.
- If a standalone script uses unexpected files, check for hard-coded constants or default paths near the top of that script.
- Some older markdown or source comments may display mojibake text in terminals with incompatible encodings; the runnable paths and option names remain usable.
