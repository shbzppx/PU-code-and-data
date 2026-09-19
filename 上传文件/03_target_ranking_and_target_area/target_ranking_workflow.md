# Target Ranking and Priority-Target Delineation Workflow

This document describes the post-processing workflow used to rank and delineate priority exploration targets in the Qixia gold ore-concentration area, Jiaodong Peninsula, eastern China. The workflow separates model-generated information from subsequent geological interpretation and target delineation.

**Package links**

- Package index: `github_code_description.md`
- Released OOF grids: `04_predictiondata/` (`data_description.md`)
- Primary-analysis protocol (including three equal-weight positive windows per deposit): `05 data_availability_metadata_preview/10_protocol_snapshot.json`

**Code extract in this folder (`03_target_ranking_and_target_area/`)**

| File | Role in this workflow |
|------|------------------------|
| `spatial_score_fusion.py` | Fuse overlapping window scores (`window_average` for released maps) |
| `model_prediction/predict.py` | Sliding-window prediction scores |
| `model_prediction/plot_labels.py` | Percentile-rank rendering (`foldrank`) |
| `model_evaluation/target_area_report.py` | Target-area / candidate summaries |
| `model_comparison/metric_protocol.py` | SR / PAR / NDI (and related) metrics |
| `model_training/spatial_cv_metrics.py` | Inner-fold fixed-area spatial metrics |

## 1. Scope and Principle

The prediction code generates spatial model-confidence or prospectivity-score distributions and the associated prediction-uncertainty information. The code does not calculate a hidden geological multi-factor score for target ranking.

Target delineation is a post-processing procedure applied to the model outputs. The final priority of a target is determined primarily from three numerical quantities:

1. the Au concentration or Au anomaly value within the candidate target;
2. the model-confidence or prospectivity score within the candidate target;
3. the prediction uncertainty within the candidate target, where lower uncertainty indicates a more stable prediction.

Fault influence strength (FIS) and the metallogenic favorability of the host strata are used for spatial delineation, geological-consistency checking, and discrimination between targets with comparable numerical evidence. They are not presented as additional model weights or as components of an undisclosed weighted sum.

## 2. Model-Generated Outputs

The prediction workflow produces a spatial score for valid grid cells in the study area. For the primary analysis, scores are obtained from the five outer-fold held-out (OOF) models and combined across folds and random-seed runs. Released Surfer grids in `04_predictiondata/` use the filename token `oof` (for example, `nnpucnn_oof_overlapBest_window_average_foldrank.grd`).

The released map products use the following operations:

- prediction scores are assigned to the valid grid cells of each outer held-out ore field by a model that did not train on that ore field;
- overlapping prediction windows are fused by `window_average`;
- within each outer fold, scores are converted to percentile ranks (`foldrank`);
- the five fold-rank surfaces are mosaicked along ore-field boundaries to form a study-area map;
- multi-seed runs (neural-network seeds listed in `05 data_availability_metadata_preview/02_training_seeds.json`) are aggregated for the released `overlapBest` products;
- the workflow also records prediction-uncertainty information used in target review.

These operations produce the numerical surfaces used by the subsequent target-ranking procedure. They do not themselves define geological target boundaries or assign final exploration priorities.

## 3. Candidate-Area Construction

Candidate areas are first extracted from the ranked model surface using a predefined prospective-area ratio (PAR) or an equivalent score threshold. In the primary analysis, the ranked high-potential area is evaluated at fixed area levels such as 5%, 10%, 20%, and 30% of the valid study area. The selected cells are grouped spatially into connected candidate regions.

The candidate-region procedure follows these steps:

1. retain valid study-area cells and their OOF mosaic scores (or percentile ranks);
2. rank the cells from high to low model score or percentile rank;
3. select the cells required by the chosen PAR or threshold rule;
4. group adjacent selected cells into connected candidate regions;
5. calculate the area, location, model-score summary, uncertainty summary, and Au summary for each region;
6. remove regions that fail the minimum mapping or reporting criteria defined for the analysis.

The PAR is a reporting and delineation rule. It is not a claim that the selected area represents the exact physical extent of mineralization.

## 4. Numerical Evidence for Priority Ranking

For each candidate region, the following summaries are calculated or extracted from the aligned spatial layers:

| Evidence | Role in priority ranking | Interpretation |
|---|---|---|
| Au concentration or Au anomaly | Primary numerical evidence | Higher Au values provide stronger geochemical support for a gold-target interpretation. |
| Model-confidence or prospectivity score | Primary numerical evidence | Higher scores indicate that the trained model assigns greater prospectivity to the candidate region. |
| Prediction uncertainty | Primary numerical evidence | Lower uncertainty indicates greater stability of the model prediction. |
| Fault influence strength (FIS) | Spatial and geological check | The candidate should show spatial consistency with the modeled influence of ore-controlling faults. |
| Metallogenic favorability of host strata | Spatial and geological check | The candidate should occur in or near strata and geological units that are favorable for gold mineralization in the study area. |

The numerical ranking gives greatest attention to the Au evidence, model confidence, and prediction stability. FIS and host-strata favorability then determine whether a high-scoring region forms a geologically coherent target and help distinguish targets with similar numerical evidence.

The workflow does not require these quantities to be collapsed into a single weighted score. When two candidate regions have similar Au support, model confidence, and uncertainty, FIS continuity and host-strata favorability provide the geological basis for the tie-break decision.

## 5. Geological Interpretation of Candidate Regions

### 5.1 Au concentration

The Au layer provides the direct geochemical evidence for gold prospectivity. For each candidate region, the analysis examines the central tendency and spatial continuity of the Au values rather than relying on a single grid cell. A candidate with a coherent Au anomaly across its mapped area receives stronger geochemical support than a candidate supported by an isolated extreme value.

### 5.2 Fault influence strength

FIS represents the spatial influence of mapped faults through the distance-based fault-control feature used by the model. Higher FIS generally indicates closer spatial association with the modeled fault system. During target interpretation, FIS is used to assess whether the candidate region follows a plausible structural corridor or lies within a structurally favorable zone.

FIS does not replace the model score and is not added to the Au value as an independent ranking weight. Its role is to check the structural coherence of the candidate boundary and to support discrimination between numerically similar candidates.

### 5.3 Metallogenic favorability of host strata

The geological-map evidence layer records the spatial distribution of lithological and stratigraphic units. The target review checks whether each candidate overlaps, or lies in a geologically plausible relationship with, host strata and units considered favorable for gold mineralization in the Qixia study area.

This geological check is interpretive. It does not imply that the target-ranking code independently estimates a stratigraphic favorability coefficient after model prediction. The geological units contribute to the trained model through the multisource feature representation and contribute to final target interpretation through spatial consistency review.

## 6. Priority Assignment

The final target priority is assigned after the candidate-region summaries and geological checks are available. The recommended decision order is:

1. compare the Au concentration or Au anomaly support across candidate regions;
2. compare the model-confidence or prospectivity-score summaries;
3. compare prediction uncertainty, giving preference to the candidate with more stable predictions when the first two quantities are comparable;
4. verify continuity with FIS and the mapped fault-controlled structural setting;
5. verify consistency with favorable host strata and the known metallogenic geological background;
6. assign the higher priority to the candidate with the stronger combined numerical evidence and clearer geological coherence.

This procedure yields a ranked list of priority exploration targets while preserving the distinction between quantitative model evidence and geological interpretation. The ranking should be reported as a decision-support result, not as a formal statistical test of differences between targets.

## 7. Released Products Used with This Workflow

- Prediction grids: `04_predictiondata/*_oof_overlapBest_window_average_foldrank.grd` (see `04_predictiondata/data_description.md`).
- Outer-fold assignment without raw coordinates:
  - `05 data_availability_metadata_preview/03_fold_manifest.csv`
  - `05 data_availability_metadata_preview/04_camp_assignment_no_xy.csv`
- Grid geometry and channel metadata: `05 data_availability_metadata_preview/01_grid_and_channels.json`
- Training seeds used for multi-seed map aggregation: `05 data_availability_metadata_preview/02_training_seeds.json`

Deposit XY coordinates, catchment-basin ID rasters, and fused training H5 cubes are not included in the public release package for licensing reasons. Exact capture-rate recomputation therefore requires restricted occurrence locations in addition to the released grids and fold tables.
