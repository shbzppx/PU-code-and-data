# Derived Spatial Data for Outer-Fold Held-Out Mineral Prospectivity Mapping

This directory is `04_predictiondata/` in the reviewer-facing code/data package.  
Package index: top-level `github_code_description.md`.  
Target-ranking narrative: `03_target_ranking_and_target_area/target_ranking_workflow.md`.

It contains the derived spatial raster products used to inspect and compare mineral prospectivity predictions in the Qixia gold ore-concentration area, Jiaodong Peninsula, eastern China, for the manuscript's revised outer-fold held-out evaluation.

The maps represent spatial generalization to an unseen ore field within the same study area. They are not results from an independent external dataset, a geographically separate validation region, or an external test campaign.

## Archive Contents

This directory contains **seven** Surfer ASCII grid (`DSAA`) files plus this description. The model-grid filenames use the lowercase `oof` token for the outer-fold held-out evaluation (five expert ore fields; one complete ore field held out per fold).

1. `nnpucnn_oof_overlapBest_window_average_foldrank.grd` — adaptive nnPU-CNN.
2. `cnn_oof_overlapBest_window_average_foldrank.grd` — conventional CNN.
3. `rf_oof_overlapBest_window_average_foldrank.grd` — random forest (RF).
4. `purf_oof_overlapBest_window_average_foldrank.grd` — positive-unlabeled random forest (PU-RF).
5. `2step_oof_overlapBest_window_average_foldrank.grd` — two-stage PU learning (Two-step PU).
6. `linear_oof_overlapBest_window_average_foldrank.grd` — linear regression baseline (Linear).
7. `ocsvm_oof_overlapBest_window_average_foldrank.grd` — one-class support vector machine (OCSVM).

**Not included in this directory (licensing / privacy):**

- catchment-basin identifier grids (e.g. watershed / SCB basin ID rasters);
- deposit coordinates (`deposit.txt` XY) or fault vertices;
- fused multi-channel H5 feature cubes used for training.

Licence-safe fold and protocol metadata without raw coordinates are provided in the companion metadata pack (`05 data_availability_metadata_preview/`), including:

- `03_fold_manifest.csv` — OOF-5 outer-fold assignment (deposit IDs per fold; no XY);
- `04_camp_assignment_no_xy.csv` — deposit-to-ore-field labels without XY;
- `01_grid_and_channels.json` — grid size, spacing, extent, and channel definitions;
- `02_training_seeds.json` — neural-network seeds used for multi-seed aggregation;
- `10_protocol_snapshot.json` — sanitized primary-analysis protocol snapshot.

Related prediction / ranking code (package extract, not a GUI app):

- `03_target_ranking_and_target_area/spatial_score_fusion.py`
- `03_target_ranking_and_target_area/model_prediction/predict.py`
- `03_target_ranking_and_target_area/model_prediction/plot_labels.py`

## Grid Specifications

The seven model-prediction grids share the following header information:

- Format: Surfer ASCII `DSAA`.
- Grid dimensions: `441` columns by `177` rows.
- X extent: `6015` to `6895` coordinate units.
- Y extent: `82532` to `82884` coordinate units.
- Nominal prediction range: `0` to `1` after map normalization/ranking preparation.
- Nominal grid spacing from the headers: `2` coordinate units in both directions.
- Physical scale (from companion metadata): `1` map unit = `50` m, so the nominal cell spacing is `100` m.

The DSAA headers do not encode a complete coordinate reference system. Users should obtain the coordinate-system definition and the physical conversion from the manuscript or the companion metadata before calculating physical areas. In particular, the coordinate spacing must not be converted to square kilometres without the appropriate metres-per-coordinate-unit value.

## Processing and Naming

The seven prediction grids were produced from outer-fold held-out predictions, overlapping-window fusion, and fold-rank aggregation:

- `oof` — outer-fold held-out mosaicking (scores in each ore field come from a model that did not train on that ore field);
- `overlapBest` — released primary overlap / seed-aggregation setting used for the manuscript comparison maps;
- `window_average` — spatial averaging of overlapping prediction windows;
- `foldrank` — within-fold percentile-rank aggregation used for the released prospectivity surface.

The released maps correspond to the revised primary analysis, including the reported `11 × 11` sliding window and `πtune = 0.5` for the weighted nnPU objective. `πtune` is a training-loss tuning coefficient and should not be confused with `πstat`, the optional KM2/other statistical prior estimates, or `rgeo`, the geological favorable-area ratio.

In the revised manuscript, the ranked outputs support five priority exploration targets. The reported top-`20%` area captures `57.14%` of the known held-out mineral occurrences. This is an observed result under the evaluated outer folds and does not represent a formal statistical test of differences between models.

## Use of the Data

The GRD files can be used to:

- display and compare the seven prospectivity surfaces;
- reproduce map-based ranking and target-area visualization when the study-area coordinate scale is available;
- inspect the spatial pattern of the outer-fold held-out predictions;
- support independent re-analysis together with the publicly released fold metadata (deposit IDs and ore-field labels without XY).

The GRD files alone are not sufficient to recompute every manuscript statistic. Exact capture-rate, Wilson interval, SRC, and target-area calculations also require mineral-occurrence locations (restricted), the outer-fold metadata, the target-area definition, and the coordinate-to-area conversion used by the analysis. This archive therefore provides derived spatial products, not a replacement for the full training input dataset.
