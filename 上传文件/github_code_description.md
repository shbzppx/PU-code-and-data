# Code and data package — contents

Reviewer-facing extract for the Qixia gold MPM study (adaptive nnPU-CNN; outer-fold held-out (OOF) evaluation).  
This file is the **package index**. It is a code/metadata extract, not a standalone installable GUI application.  

Companion narrative documents:

- `03_target_ranking_and_target_area/target_ranking_workflow.md` — target ranking / delineation
- `04_predictiondata/data_description.md` — released OOF grids
- `05 data_availability_metadata_preview/README.txt` — metadata inventory

## Top-level layout

| Path | Contents |
|------|----------|
| `github_code_description.md` | This contents list |
| `01_sampling_and_spatial_split/` | Sampling, spatial split, nested fault calibration |
| `02_adaptive_pu_loss_and_km2/` | Adaptive nnPU loss and KM2 prior estimation |
| `03_target_ranking_and_target_area/` | Prediction scores, ranking metrics, target-area reporting |
| `04_predictiondata/` | Released OOF prospectivity grids + data note |
| `05 data_availability_metadata_preview/` | Licence-safe metadata (seeds, folds, protocol, Word2Vec list) |

## Key documents (by location)

| Document | Location |
|----------|----------|
| Target-ranking workflow | `03_target_ranking_and_target_area/target_ranking_workflow.md` |
| Zenodo / prediction-grid description | `04_predictiondata/data_description.md` |
| Metadata pack inventory | `05 data_availability_metadata_preview/README.txt` |
| Grid & channels | `05 data_availability_metadata_preview/01_grid_and_channels.json` |
| Training seeds | `05 data_availability_metadata_preview/02_training_seeds.json` |
| OOF-5 fold manifest | `05 data_availability_metadata_preview/03_fold_manifest.csv` |
| Deposit–ore-field table (no XY) | `05 data_availability_metadata_preview/04_camp_assignment_no_xy.csv` |
| KM2 settings | `05 data_availability_metadata_preview/05_km2_settings.json` |
| Word2Vec unit map | `05 data_availability_metadata_preview/06_word2vec_unit_map.csv` |
| Word2Vec hyperparameters | `05 data_availability_metadata_preview/07_word2vec_summary_public.json` |
| Word2Vec corpus bibliography (English) | `05 data_availability_metadata_preview/08_word2vec_references_English.md` |
| Word2Vec corpus bibliography (Chinese, sorted) | `05 data_availability_metadata_preview/09_word2vec_references_sorted.md` |
| Protocol snapshot | `05 data_availability_metadata_preview/10_protocol_snapshot.json` |

## Code files (current extract)

### `01_sampling_and_spatial_split/`

| File | Role |
|------|------|
| `model_training/dataset.py` | Feature loading, positive/unlabeled windows, sampling |
| `model_training/config.py` | CLI / default hyperparameters |
| `model_training/pu_batch_sampler.py` | PU mini-batches with positives |
| `model_training/spatial_validation_splits.py` | Inner leave-one-ore-field splits |
| `model_comparison/spatial_mineral_splitter.py` | Deposit → ore-field grouping |
| `model_comparison/spatial_region_splitter.py` | Catchment / region isolation |
| `support/nested_fault_calibration.py` | Fold-wise FIS λ calibration |
| `support/reviewer_protocol.py` | Primary-protocol helpers |
| `support/feature_channel_utils.py` | Channel selection utilities |

### `02_adaptive_pu_loss_and_km2/`

| File | Role |
|------|------|
| `model_training/pu_loss.py` | Standard / adaptive nnPU loss |
| `model_training/nnpu_stable_adaptive_pu_loss.py` | nnPU-CNN stable adaptive wrapper |
| `model_training/prior_estimation.py` | KM2 (and related) prior estimators |
| `model_training/utils.py` | Loss factory / training utilities |
| `model_training/train_utils.py` | Train step; epoch-end adaptive γ update |
| `model_training/cross_validation.py` | Inner CV / spatial early-stopping helpers |

### `03_target_ranking_and_target_area/`

| File | Role |
|------|------|
| `target_ranking_workflow.md` | Narrative target-ranking / delineation workflow |
| `spatial_score_fusion.py` | Window-average (and related) score fusion |
| `model_prediction/predict.py` | Sliding-window prediction |
| `model_prediction/plot_labels.py` | Percentile-rank map rendering |
| `model_evaluation/target_area_report.py` | Target-area reporting |
| `model_comparison/metric_protocol.py` | SR / PAR / NDI (and related) metrics |
| `model_training/spatial_cv_metrics.py` | Inner-fold fixed-area spatial metrics |

### `04_predictiondata/`

| File | Role |
|------|------|
| `nnpucnn_oof_overlapBest_window_average_foldrank.grd` | Adaptive nnPU-CNN |
| `cnn_oof_overlapBest_window_average_foldrank.grd` | CNN |
| `rf_oof_overlapBest_window_average_foldrank.grd` | RF |
| `purf_oof_overlapBest_window_average_foldrank.grd` | PU-RF |
| `2step_oof_overlapBest_window_average_foldrank.grd` | Two-step PU |
| `linear_oof_overlapBest_window_average_foldrank.grd` | Linear |
| `ocsvm_oof_overlapBest_window_average_foldrank.grd` | OCSVM |
| `data_description.md` | Grid archive description |

## Mapping to Data/Code Availability items

| Reviewer item | Where to find it in this package |
|---------------|----------------------------------|
| Adaptive loss implementation | `02_.../pu_loss.py`, `nnpu_stable_adaptive_pu_loss.py` |
| OOF partitions | `05_.../03_fold_manifest.csv`, `04_camp_assignment_no_xy.csv` |
| Sampling procedure | `01_.../dataset.py`, `pu_batch_sampler.py`; protocol in `05_.../10_protocol_snapshot.json` |
| Random seeds | `05_.../02_training_seeds.json` |
| KM2 settings | `05_.../05_km2_settings.json`; code in `02_.../prior_estimation.py` |
| Target-ranking workflow | `03_.../target_ranking_workflow.md` + ranking/prediction code above |
| Processed prediction maps | `04_predictiondata/*_oof_*.grd` |
| Word2Vec document list | `05_.../08_word2vec_references_English.md` (preferred); `09_..._sorted.md` Chinese mirror |
