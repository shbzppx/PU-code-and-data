import argparse
import os

import torch

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AUX_DIR = os.path.join(_CODE_ROOT, "训练辅助数据")
_FEATURE_DIR = os.path.join(_CODE_ROOT, "新版本数据", "训练辅助数据")
_DEFAULT_DATASET = os.path.join(_FEATURE_DIR, "A2K_main_dim5_seed42_faultfull.h5")
_DEFAULT_LABEL = os.path.join(_AUX_DIR, "deposit.txt")
_DEFAULT_FAULT_LINES = os.path.join(_AUX_DIR, "断裂构造_vertices.csv")
_DEFAULT_BASIN_GRD = os.path.join(_AUX_DIR, "汇水域.grd")
_DEFAULT_CAMP_ASSIGN = os.path.join(_AUX_DIR, "矿田分布.csv")


def get_args_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="PU learning PyTorch implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--batchsize", "-b", type=int, default=32, help="Mini batch size")
    parser.add_argument(
        "--nnpucnn-positives-per-batch",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="Positive examples in each nnPU-CNN training batch, independent of batch size",
    )
    parser.add_argument(
        "--nnpucnn-unlabeled-coverage",
        type=float,
        default=0.5,
        help="Target fraction of unique unlabeled examples covered per nnPU-CNN epoch (0, 1]",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help="Path to the feature H5 file",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default=_DEFAULT_LABEL,
        help="Path to the mineral coordinate TXT or label H5 file",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="spatial_cluster_holdout_cv",
        choices=[
            "legacy",
            "spatial_cluster",
            "spatial_hard",
            "spatial_stratified",
            "spatial_cluster_holdout_cv",
            "leave_one_camp",
            "leave_one_fault",
            "variogram_block_cv",
        ],
        help="Dataset split mode for the training panel",
    )
    parser.add_argument(
        "--leave-one-camp-index",
        type=int,
        default=0,
        help="Which KMeans camp/cluster is held out as external test for leave_one_camp",
    )
    parser.add_argument(
        "--leave-one-fault-id",
        type=str,
        default="",
        help="Fault ID held out as external test for leave_one_fault (e.g. F1); empty uses first fault",
    )
    parser.add_argument(
        "--deposit-fault-assignment",
        type=str,
        default="",
        help="CSV with deposit_id,x,y,fault_id for leave_one_fault; empty auto-assigns nearest fault",
    )
    parser.add_argument(
        "--deposit-camp-assignment",
        type=str,
        default=_DEFAULT_CAMP_ASSIGN if os.path.exists(_DEFAULT_CAMP_ASSIGN) else "",
        help="CSV with deposit_id,x,y,camp_id for expert-confirmed leave_one_camp groups",
    )
    parser.add_argument(
        "--expert-camp-confirmed",
        action="store_true",
        default=False,
        help="Confirm that --deposit-camp-assignment was reviewed by a geological expert",
    )
    parser.add_argument(
        "--expert-fault-confirmed",
        action="store_true",
        default=False,
        help="Confirm that --deposit-fault-assignment was reviewed by a geological expert",
    )
    parser.add_argument(
        "--reviewer-protocol",
        action="store_true",
        default=False,
        help="Enforce reviewer-ready distance, grouping, prior and split checks",
    )
    parser.add_argument(
        "--meters-per-coordinate-unit",
        type=float,
        default=None,
        help=(
            "Physical scale of input map coordinates. Reviewer workflows require this value "
            "unless authoritative H5 metadata provides it; no 50/100 m assumption is made."
        ),
    )
    parser.add_argument(
        "--basin-grd-path",
        type=str,
        default=_DEFAULT_BASIN_GRD if os.path.exists(_DEFAULT_BASIN_GRD) else "",
        help="Catchment/basin DSAA .grd; when set, leave-one-camp/fault unlabeled split keeps whole basins (no cutting)",
    )
    parser.add_argument(
        "--fault-lines-path",
        type=str,
        default=_DEFAULT_FAULT_LINES if os.path.exists(_DEFAULT_FAULT_LINES) else "",
        help="Fault polyline CSV/TXT with columns x,y,fault_id (or x1,y1,x2,y2,fault_id)",
    )
    parser.add_argument(
        "--nested-fault-calibration",
        action="store_true",
        default=False,
        help="OGR R1.5: calibrate fault exponential-decay length inside each train fold only",
    )
    parser.add_argument(
        "--fault-decay-quantile",
        type=float,
        default=0.8,
        help="Quantile of train-deposit→fault distances used as decay length λ "
        "(e.g. 0.8≈80%%, 0.9≈90%% of train deposits fall within stronger decay); range (0,1]",
    )
    parser.add_argument(
        "--fault-ablation",
        type=str,
        default="",
        choices=[
            "",
            "fault_full",
            "fault_decay_only",
            "fault_distance_only",
            "fault_intersection_only",
            "fault_none",
        ],
        help="OGR R1.5 fault-feature ablation preset (applied via selected-channels if not set)",
    )
    parser.add_argument(
        "--variogram-range-m",
        type=float,
        default=2000.0,
        help="Variogram block edge length in metres; converted with the input distance scale",
    )
    parser.add_argument(
        "--fail-on-single-variogram-block",
        action="store_true",
        default=False,
        help="Raise if variogram_block_cv collapses to a single mineral-occupied block",
    )
    parser.add_argument(
        "--positive-window-mode",
        type=str,
        default="three_windows_equal_weight",
        choices=[
            "three_windows_equal_weight",
            "multi_window",
            "one_window_per_unit",
            "multi_window_weighted",
            "multi_window_distance_weighted",
            "multi_window_weighted_hw5",
            "multi_window_distance_weighted_hw5",
        ],
        help=(
            "Positive windows (manuscript default three_windows_equal_weight): "
            "three_windows_equal_weight=1 mineral-centered patch + 2 nearest Chebyshev "
            "neighbors, deposit-level loss 1/3; "
            "one_window_per_unit=mineral-centered nearest patch; "
            "multi_window=all containing sliding windows; "
            "multi_window_weighted=fixed neighborhood (default refW=5 → ±2 cells; "
            "1 center + random) + deposit-level loss weight 1/n; "
            "multi_window_distance_weighted=same ±2 sampling + Gaussian distance-decay "
            "loss weights (σ default 1), renormalized per deposit; "
            "multi_window_weighted_hw5 / multi_window_distance_weighted_hw5="
            "same 1+4 scheme with ±5 cells (σ default 2.5 for distance); "
            "feature patch still uses W"
        ),
    )
    parser.add_argument(
        "--multi-window-max-per-unit",
        type=int,
        default=5,
        help=(
            "For capped multi-window modes: max positive windows per mineral "
            "(three_windows_equal_weight always uses 3 = 1 center + 2 nearest; "
            "legacy ±2/±5 modes: 1 center + up to max-1 random neighbors; default 5)"
        ),
    )
    parser.add_argument(
        "--multi-window-sample-ref-window",
        type=int,
        default=5,
        help=(
            "For capped multi-window modes: reference window that anchors the fixed "
            "sampling neighborhood (default 5 → Chebyshev halfwidth 2). Keeps "
            "positive support comparable across W=5..17."
        ),
    )
    parser.add_argument(
        "--multi-window-sample-halfwidth",
        type=int,
        default=None,
        help=(
            "Optional override of fixed neighborhood halfwidth in grid cells; "
            "if omitted, uses floor(ref_window/2)"
        ),
    )
    parser.add_argument(
        "--multi-window-distance-sigma",
        type=float,
        default=None,
        help=(
            "Gaussian σ (Chebyshev grid cells) for distance-weighted capped multi-window "
            "modes; default 1.0 for ±2 (multi_window_distance_weighted) and 2.5 for ±5 "
            "(multi_window_distance_weighted_hw5)"
        ),
    )
    parser.add_argument(
        "--deposit-loss-weighting",
        action="store_true",
        default=False,
        help="Weight positive PU loss by deposit unit so multi-window deposits do not dominate",
    )
    parser.add_argument(
        "--merge-distance-m",
        type=float,
        default=0.0,
        help="Mineralization-unit merge distance in metres; 0 keeps each deposit independent",
    )
    parser.add_argument(
        "--metric-unit",
        type=str,
        default="deposit_unit",
        choices=["deposit_unit", "window"],
        help="Primary evaluation unit; deposit_unit is required for OGR revision reporting",
    )
    parser.add_argument(
        "--prior-mode",
        type=str,
        default="manual",
        choices=["manual", "grid", "estimate_en", "estimate_km2", "auto"],
        help="Class-prior source: manual/grid tuning, Elkan-Noto estimate, KM2 estimate, or window auto ratio",
    )
    parser.add_argument(
        "--geological-favorable-area-ratio",
        type=float,
        default=None,
        help="Independent geological favorable-area ratio for comparison with statistical/tuning priors",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Repeated-run count used by fairness/significance protocol",
    )
    parser.add_argument(
        "--reviewer-primary-protocol",
        action="store_true",
        default=False,
        help=(
            "Enforce the primary analysis: all expert-group outer folds, one window per "
            "mineralization unit, seed grid matched by n_repeats, inner CV and validation-only early stopping"
        ),
    )
    parser.add_argument(
        "--unlabeled-contamination-scope",
        type=str,
        default="train_minerals",
        choices=["train_minerals", "all_minerals"],
        help=(
            "Which deposits carve windows out of the unlabeled pool under mineral-centered "
            "positive construction. train_minerals (default / revised): only modeling deposits; "
            "held-out neighborhoods stay available for the outer test ranking domain. "
            "all_minerals: August/CCC-style; held-out neighborhoods are also excluded from unlabeled."
        ),
    )
    parser.add_argument(
        "--fairness-protocol",
        action="store_true",
        default=True,
        help="Record and enforce shared fairness protocol metadata for baseline comparisons",
    )
    parser.add_argument(
        "--no-fairness-protocol",
        action="store_false",
        dest="fairness_protocol",
        help="Disable fairness protocol metadata enforcement",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.0,
        help="Proportion of data reserved for validation/test split (0-1)",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Fraction of data used after stratified sampling (0-1]",
    )
    parser.add_argument(
        "--sample-ratio-grid",
        type=str,
        default="",
        help="Comma-separated development sampling ratios for grid experiments, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--selected-channels-grid",
        type=str,
        default="",
        help="Semicolon-separated zero-based channel groups for grid experiments, e.g. all;0,1,2;3,4,5",
    )
    parser.add_argument(
        "--buffer-radius",
        type=float,
        default=0.0,
        help=(
            "Positive-halo exclusion radius in metres around deposits when building unlabeled "
            "(gray-zone removal; not labeled as positives). "
            "OGR M/N/P UI defaults to 500; pass 0 to disable"
        ),
    )
    parser.add_argument("--spatial-cluster-n-clusters", type=int, default=4, help="Number of KMeans clusters for spatial splitting")
    parser.add_argument("--spatial-cluster-train-ratio", type=float, default=0.8, help="Train ratio inside each spatial cluster")
    parser.add_argument("--full-mineral-training", action="store_true", help="Use all mineral points for final training without an external mineral test split")
    parser.add_argument(
        "--mineral-training-strategy",
        type=str,
        default="holdout_cv",
        choices=["holdout", "train_val", "all_minerals", "holdout_cv"],
        help=(
            "Final mineral training strategy for spatial mineral mode: "
            "holdout keeps the internal validation split, train_val trains on all dev train+val minerals while keeping external test minerals, "
            "all_minerals trains with all minerals and leaves no external mineral test split, "
            "holdout_cv keeps a per-cluster external test split and runs spatial CV inside the development set."
        ),
    )
    parser.add_argument(
        "--spatial-cv-buffer-distance",
        type=float,
        default=500.0,
        help=(
            "Train/val (or train/test) isolation buffer in metres; converted using input "
            "metadata. Manuscript inner leave-one-camp default is 500 m along catchment "
            "boundaries. 0 disables the strip."
        ),
    )
    parser.add_argument(
        "--spatial-metric-threshold-strategy",
        type=str,
        default="max_ei",
        choices=["max_ei", "fixed", "fixed_area"],
        help=(
            "Threshold strategy for SR/PAF/EI: max_ei scans score thresholds; "
            "fixed uses --spatial-metric-fixed-threshold; "
            "fixed_area uses top-k cells by score at --spatial-metric-area-fractions"
        ),
    )
    parser.add_argument("--spatial-metric-threshold-step", type=float, default=0.001, help="Threshold step for spatial CV SR/PAF/EI evaluation")
    parser.add_argument("--spatial-metric-distance-threshold", type=float, default=100.0, help="Primary distance-tolerant mineral hit threshold in metres; sensitivity-only and never used for model selection")
    parser.add_argument(
        "--spatial-metric-distance-sensitivity-thresholds",
        type=str,
        default="250",
        help="Comma-separated additional distance-tolerant SR thresholds in metres; reported as sensitivity only",
    )
    parser.add_argument("--spatial-metric-fixed-threshold", type=float, default=None, help="Optional fixed threshold for spatial CV metrics; default scans thresholds and maximizes EI")
    parser.add_argument(
        "--spatial-metric-area-fractions",
        type=str,
        default="",
        help="Comma-separated prospective-area budgets for fixed_area strategy, e.g. 0.05,0.10,0.20,0.30 or 5,10,20,30",
    )
    parser.add_argument(
        "--spatial-metric-primary-area-fraction",
        type=float,
        default=0.05,
        help="Primary area budget used for cv_sr/cv_paf/cv_ei summary under fixed_area (default 0.05)",
    )
    parser.add_argument(
        "--tie-aware-center-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefer tie-expected nearest-cell EI for inner hyperparameter selection "
            "(default on). Use --no-tie-aware-center-selection to fall back to legacy EI."
        ),
    )
    parser.add_argument(
        "--spatial-checkpoint-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Within CV folds, checkpoint/early-stop by validation fixed-area "
            "tie-expected EI when spatial metrics are available (default on)."
        ),
    )
    parser.add_argument(
        "--spatial-checkpoint-every",
        type=int,
        default=1,
        help="Compute validation spatial checkpoint metrics every N epochs (default 1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay (L2) for neural optimizers (default 1e-4)",
    )
    parser.add_argument(
        "--adaptive-gamma-ema",
        type=float,
        default=0.8,
        help="EMA momentum for AdaptivePULoss epoch-level gamma updates (default 0.8)",
    )
    parser.add_argument(
        "--inner-screening-only",
        action="store_true",
        help="Run reviewer inner candidate selection only; skip final refit and outer holdout evaluation",
    )
    parser.set_defaults(nested_save_all_candidate_weights=True)
    parser.add_argument(
        "--nested-screening-only-weights",
        dest="nested_save_all_candidate_weights",
        action="store_false",
        help=(
            "Legacy slim nested grid for leave-one-camp / random-one-camp inner modes: "
            "inner-screen all π×window candidates and write model.pth only for the inner-CV winner. "
            "Default is to save weights for every π×window while still marking hp_selected from inner metrics."
        ),
    )
    parser.add_argument(
        "--footprint-reference-patch-size",
        type=int,
        default=0,
        help=(
            "Odd reference window for cross-fold feature-footprint embargo. "
            "0 (default) disables the embargo; positive odd size enables it "
            "(e.g. 17 = 16 grid intervals / ±1600 m on a 100 m grid)."
        ),
    )
    parser.add_argument(
        "--inner-camp-cv-mode",
        type=str,
        default="leave_one_camp",
        help=(
            "Inner CV among modeling camps for leave-one-camp: "
            "leave_one_camp (manuscript: nested OOF among remaining 4), "
            "random_one_camp_val (appendix: single random val camp), "
            "all_internal_train_select (no inner camp CV; select π/window by train metrics), "
            "all_internal_outer_select (no inner camp CV; optimistic select by outer test). "
            "Empty = packed blocks."
        ),
    )
    parser.add_argument(
        "--report-validation-tau-sensitivity",
        action="store_true",
        help="Under fixed_area primary, also report max-EI validation τ sensitivity on folds and outer test",
    )
    parser.add_argument(
        "--no-report-validation-tau-sensitivity",
        action="store_true",
        help="Disable validation-τ sensitivity sidecar",
    )
    parser.add_argument("--spatial-random-state", type=int, default=19, help="Random seed for spatial splitting")
    parser.add_argument(
        "--spatial-random-state-grid",
        type=str,
        default="",
        help="Comma-separated spatial random seeds for grid experiments, e.g. 19,42,43",
    )
    parser.add_argument("--no-ore-path", type=str, default=None, help="Optional no-ore coordinate file for spatial mode")
    parser.add_argument("--augmentation-enabled", action="store_true", help="Enable small numeric perturbation on training samples only")
    parser.add_argument("--augmentation-noise-std", type=float, default=0.05, help="Noise strength for training sample augmentation")
    parser.add_argument("--epoch", "-e", default=50, type=int, help="# of epochs to learn")
    parser.add_argument(
        "--eval-monitor-max-samples",
        type=int,
        default=2048,
        help="Max samples for per-epoch test/val monitoring curves; 0=use full set. Speeds A2/basin leave-one when test is huge.",
    )
    parser.add_argument("--beta", "-B", default=0, type=float, help="Beta parameter of nnPU")
    parser.add_argument("--gamma", "-G", default=1, type=float, help="Gamma parameter of nnPU")
    parser.add_argument(
        "--gamma-grid",
        type=str,
        default="",
        help="Comma-separated gamma values for grid experiments, e.g. 0.5,1,2,5",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="pucnn",
        choices=[
            "linear",
            "3lp",
            "mlp",
            "cnn",
            "cnnt",
            "cntt",
            "pucnn",
            "nnpucnn",
            "rnapucnn",
            "rncapucnn",
            "rncpucnn",
            "rnfcapucnn",
            "rnfcspucnn",
            "rngapucnn",
            "rngspucnn",
            "rnlrapucnn",
            "rnlrspucnn",
            "rnscapucnn",
            "rnscspucnn",
            "pucnnt",
            "pucnntransformer",
            "rf",
            "purf",
            "ocsvm",
            "2step",
        ],
        help="The name of a classification model",
    )
    parser.add_argument(
        "--model-grid",
        type=str,
        default="",
        help="Comma-separated model names for grid experiments, e.g. cnn,cnnt,pucnntransformer",
    )
    parser.add_argument("--stepsize", "-s", default=1e-5, type=float, help="Stepsize of gradient method")
    parser.add_argument("--out", "-o", default="result", help="Directory to output the result")
    parser.add_argument(
        "--resume-grid-dir",
        type=str,
        default="",
        help="Reuse an existing grid_training_* directory; skip combos already marked completed in grid_summary.json",
    )
    parser.add_argument("--auto-prior", action="store_true", default=False, help="Automatically calculate prior probability")
    parser.add_argument("--no-auto-prior", action="store_false", dest="auto_prior", help="Disable automatic prior calculation")
    parser.add_argument("--manual-prior", type=float, default=0.2, help="Manually set prior probability")
    parser.add_argument(
        "--prior-grid",
        type=str,
        default="",
        help="Comma-separated manual prior probabilities for grid experiments, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument("--img_size", default=16, type=int, help="Input image size")
    parser.add_argument(
        "--patch-size",
        default=9,
        type=int,
        help="Sliding window size for raw H5 inputs",
    )
    parser.add_argument(
        "--patch-size-grid",
        type=str,
        default="",
        help="Comma-separated patch/window sizes for grid experiments, e.g. 7,9,11",
    )
    parser.add_argument(
        "--patch-stride",
        default=1,
        type=int,
        help="Sliding window stride for raw H5 inputs",
    )
    parser.add_argument(
        "--reflect-padding",
        action="store_true",
        default=True,
        help="Enable reflect padding when generating raw-H5 sliding windows",
    )
    parser.add_argument(
        "--no-reflect-padding",
        action="store_false",
        dest="reflect_padding",
        help="Disable reflect padding when generating raw-H5 sliding windows",
    )
    parser.add_argument(
        "--selected-channels",
        type=str,
        default="",
        help="Comma-separated zero-based channel indices to keep; empty means all channels",
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["standard", "adaptive"],
        default="adaptive",
        help="Type of PU loss function to use",
    )
    parser.add_argument(
        "--loss-type-grid",
        type=str,
        default="",
        help="Comma-separated PU loss types for grid experiments, e.g. standard,adaptive",
    )
    parser.add_argument(
        "--adaptive-window",
        type=int,
        default=10,
        help="Window size for adaptive PU loss",
    )
    parser.add_argument(
        "--adaptive-lambda",
        type=float,
        default=1.0,
        help="Strength for adaptive gamma adjustment in adaptive PU loss",
    )
    parser.add_argument(
        "--adaptive-gamma-min",
        type=float,
        default=None,
        help="Lower clipping bound for adaptive gamma; default uses the base gamma",
    )
    parser.add_argument(
        "--adaptive-gamma-max",
        type=float,
        default=None,
        help="Upper clipping bound for adaptive gamma; default uses max(base gamma, 5)",
    )
    parser.add_argument(
        "--adaptive-per-batch",
        action="store_true",
        help="V17 PU-CNN: adapt gamma every training batch instead of once per epoch (EMA)",
    )
    parser.add_argument("--rn-warmup-epochs", type=int, default=5, help="RN-aNNPU warm-up epochs")
    parser.add_argument("--rn-fraction", type=float, default=0.25, help="Lowest-scoring U fraction used as reliable negatives")
    parser.add_argument("--rn-base-weight", type=float, default=0.10, help="Base reliable-negative BCE weight")
    parser.add_argument("--rn-rank-weight", type=float, default=0.05, help="Positive-vs-RN ranking loss weight")
    parser.add_argument("--rn-rank-margin", type=float, default=0.5, help="Positive-vs-RN ranking margin in logit space")
    parser.add_argument("--rn-adaptive-window", type=int, default=10, help="Correction-frequency window for adaptive RN weight")
    parser.add_argument("--rn-max-weight", type=float, default=0.30, help="Maximum reliable-negative BCE weight")
    parser.add_argument("--rn-teacher-momentum", type=float, default=0.99, help="EMA teacher momentum for confidence RN models")
    parser.add_argument("--rn-min-confidence", type=float, default=0.55, help="Minimum teacher negative confidence before adaptive RN weight grows")
    parser.add_argument("--rn-quality-momentum", type=float, default=0.90, help="EMA momentum for RN confidence/ranking quality")
    parser.add_argument("--rn-need-target", type=float, default=0.20, help="Violation rate at which conservative RN assistance reaches full need")
    parser.add_argument("--rn-min-weight-ratio", type=float, default=0.50, help="Minimum conservative adaptive multiplier at trusted zero-need state")
    parser.add_argument("--rn-max-weight-ratio", type=float, default=1.25, help="Maximum conservative adaptive multiplier before the absolute RN cap")

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=4,
        help="Number of folds for inner cross-validation (manuscript leave-one among remaining 4 camps)",
    )
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stratified k-fold cross-validation (use --no-stratified to disable)",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use inner-validation early stopping (use --no-early-stopping to disable)",
    )
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.set_defaults(train_loss_plateau_es=True)
    parser.add_argument(
        "--train-loss-plateau-es",
        dest="train_loss_plateau_es",
        action="store_true",
        help="Enable train-loss plateau early stopping (no test/held-out monitoring)",
    )
    parser.add_argument(
        "--no-train-loss-plateau-es",
        dest="train_loss_plateau_es",
        action="store_false",
        help="Disable train-loss plateau early stopping",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=10,
        help="Patience (epochs without relative train-loss improvement) for plateau early stopping",
    )
    parser.add_argument(
        "--plateau-min-epochs",
        type=int,
        default=20,
        help="Minimum epochs before train-loss plateau early stopping can trigger",
    )
    parser.add_argument(
        "--plateau-delta",
        type=float,
        default=1e-3,
        help="Relative train-loss improvement threshold for plateau early stopping",
    )
    parser.set_defaults(
        save_training_curve_artifacts=False,
        save_fold_weight_artifacts=False,
        save_inner_hyperparameter_search=False,
        cleanup_ephemeral_artifacts=True,
    )
    parser.add_argument(
        "--save-training-curve-artifacts",
        dest="save_training_curve_artifacts",
        action="store_true",
        help="Also save learning/CV curve PNGs and training_history (off by default to save disk)",
    )
    parser.add_argument(
        "--save-fold-weight-artifacts",
        dest="save_fold_weight_artifacts",
        action="store_true",
        help="Keep per-fold model_fold*.pth under m/ and ihp (off by default; final uses best_model)",
    )
    parser.add_argument(
        "--save-inner-hyperparameter-search",
        dest="save_inner_hyperparameter_search",
        action="store_true",
        help="Write inner_hyperparameter_search.json process dump (off by default)",
    )
    parser.add_argument(
        "--no-cleanup-ephemeral-artifacts",
        dest="cleanup_ephemeral_artifacts",
        action="store_false",
        help="Do not delete ephemeral plots/caches/ihp weights after each combo finishes",
    )

    return parser
