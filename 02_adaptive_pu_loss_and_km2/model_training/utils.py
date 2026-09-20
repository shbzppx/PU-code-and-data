import os
import sys
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pu_loss import PULoss, AdaptivePULoss
from model import LinearClassifier, ThreeLayerPerceptron, MultiLayerPerceptron, CNN, CNNTransformer, CNNTokenTransformer, OneClassSVMClassifier, PURandomForestClassifier, RandomForestBinaryClassifier, TwoStepPULearning

SUPERVISED_NEURAL_MODELS = {"linear", "3lp", "mlp", "cnn", "cnnt", "cntt"}
PU_NEURAL_MODELS = {
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
}
NON_NEURAL_MODELS = {"rf", "ocsvm", "2step", "purf"}
IMAGE_MODEL_KEYS = {
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
}
POSITIVE_BATCH_PU_MODELS = {
    "nnpucnn",
    "rnapucnn",
    "rncapucnn",
    "rncpucnn",
    "rngapucnn",
    "rngspucnn",
    "rnlrapucnn",
    "rnlrspucnn",
    "rnscapucnn",
    "rnscspucnn",
}


class SupervisedBCEWithLogitsLoss:
    """Binary supervised loss while keeping the project's 1/-1 label format.

    ``pos_weight`` follows PyTorch BCEWithLogitsLoss: weight for the positive class,
    typically n_neg / n_pos so rare positives are not dominated by abundant
    unlabeled-as-negative samples (aligned with RF class_weight='balanced').
    """

    def __init__(self, pos_weight=None):
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def __call__(self, inp, target, sample_weights=None):
        logits = inp.view(-1)
        labels = (target.view(-1) > 0).float().to(device=logits.device, dtype=logits.dtype)
        pos_weight = None
        if self.pos_weight is not None and np.isfinite(self.pos_weight) and self.pos_weight > 0:
            pos_weight = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        if sample_weights is None:
            return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        weights = sample_weights.view(-1).to(device=logits.device, dtype=logits.dtype)
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none"
        )
        denom = torch.clamp(weights.sum(), min=1e-8)
        return (per_sample * weights).sum() / denom


def compute_supervised_pos_weight(labels):
    """Return n_neg/n_pos from 1/-1 (or 1/0) labels; 1.0 if either class is empty."""
    if labels is None:
        return 1.0
    if torch.is_tensor(labels):
        y = labels.detach().cpu().numpy().reshape(-1)
    else:
        y = np.asarray(labels).reshape(-1)
    if len(y) == 0:
        return 1.0
    n_pos = int(np.sum(y > 0))
    n_neg = int(np.sum(y <= 0))
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def is_pu_neural_model(model_name):
    return model_name in PU_NEURAL_MODELS


def is_non_neural_model(model_name):
    return model_name in NON_NEURAL_MODELS


def is_image_model(model_name):
    return model_name in IMAGE_MODEL_KEYS


def seed_training_randomness(seed, deterministic=True):
    """Seed model initialization, loaders, NumPy and CUDA for one run."""
    seed = int(seed)
    deterministic = bool(deterministic)
    if deterministic:
        # Required by CUDA >= 10.2 for deterministic cuBLAS matrix operations.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda": bool(torch.cuda.is_available()),
        "cudnn_deterministic": deterministic,
        "deterministic_algorithms": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def select_model(model_name):
    """根据模型名称选择对应的模型类"""
    if model_name == 'linear':
        from model.linear import LinearClassifier # <- 修改这里
        return LinearClassifier
    elif model_name == '3lp':
        from model.mlp import ThreeLayerPerceptron
        return ThreeLayerPerceptron
    elif model_name == 'mlp':
        from model.mlp import MultiLayerPerceptron
        return MultiLayerPerceptron
    elif model_name == 'cnn':
        from model.cnn import CNN
        return CNN
    elif model_name == 'cnnt':
        from model.cnn_transformer import CNNTransformer
        return CNNTransformer
    elif model_name == 'cntt':
        from model.cnn_token_transformer import CNNTokenTransformer
        return CNNTokenTransformer
    elif model_name == 'pucnn':
        from model.pu_cnn import PUCNN
        return PUCNN
    elif model_name == 'nnpucnn':
        from model.nnpu_cnn import NNPUCNN
        return NNPUCNN
    elif model_name == 'rnapucnn':
        from model.rn_pu_cnn import RNAdaptivePUCNN
        return RNAdaptivePUCNN

    elif model_name == 'rncapucnn':
        from model.rn_confidence_pu_cnn import RNConfidenceAdaptivePUCNN
        return RNConfidenceAdaptivePUCNN
    elif model_name == 'rncpucnn':
        from model.rn_confidence_pu_cnn import RNConfidenceStaticPUCNN
        return RNConfidenceStaticPUCNN
    elif model_name == 'rnfcapucnn':
        from model.rn_full_coverage_pu_cnn import RNFullCoverageAdaptivePUCNN
        return RNFullCoverageAdaptivePUCNN
    elif model_name == 'rnfcspucnn':
        from model.rn_full_coverage_pu_cnn import RNFullCoverageStaticPUCNN
        return RNFullCoverageStaticPUCNN
    elif model_name == 'rngapucnn':
        from model.rn_gap_pu_cnn import RNGapAdaptivePUCNN
        return RNGapAdaptivePUCNN
    elif model_name == 'rngspucnn':
        from model.rn_gap_pu_cnn import RNGapStaticPUCNN
        return RNGapStaticPUCNN
    elif model_name == 'rnlrapucnn':
        from model.rn_logit_rank_pu_cnn import RNLogitRankAdaptivePUCNN
        return RNLogitRankAdaptivePUCNN
    elif model_name == 'rnlrspucnn':
        from model.rn_logit_rank_pu_cnn import RNLogitRankStaticPUCNN
        return RNLogitRankStaticPUCNN
    elif model_name == 'rnscapucnn':
        from model.rn_spatial_conservative_pu_cnn import RNSpatialConservativeAdaptivePUCNN
        return RNSpatialConservativeAdaptivePUCNN
    elif model_name == 'rnscspucnn':
        from model.rn_spatial_conservative_pu_cnn import RNSpatialConservativeStaticPUCNN
        return RNSpatialConservativeStaticPUCNN
    elif model_name == 'pucnnt':
        from model.cnn_transformer import CNNTransformer
        return CNNTransformer
    elif model_name == 'pucnntransformer':
        from model.cnn_token_transformer import CNNTokenTransformer
        return CNNTokenTransformer
    elif model_name == 'rf':
        from model.random_forest import RandomForestBinaryClassifier
        return RandomForestBinaryClassifier
    elif model_name == 'ocsvm':
        from model.one_class_svm import OneClassSVMClassifier
        return OneClassSVMClassifier
    elif model_name == 'purf':
        from model.pu_random_forest import PURandomForestClassifier
        return PURandomForestClassifier
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def instantiate_model(model_name, prior, input_shape):
    """根据输入形状创建模型，尽可能传递 (channels, height, width)。"""
    channels, height, width = input_shape
    input_dim = channels * height * width
    model_cls = select_model(model_name)
    try:
        return model_cls(prior, input_dim, input_shape=input_shape)
    except TypeError:
        # 回退到旧签名
        return model_cls(prior, input_dim)

def create_loss_function(args, prior, labels=None):
    """根据参数创建损失函数。

    For supervised neural baselines (CNN/linear/...), ``labels`` should be the
    fold-train 1/-1 labels so pos_weight = n_neg/n_pos matches RF balanced weighting.
    """
    if not is_pu_neural_model(args.model):
        pos_weight = compute_supervised_pos_weight(labels)
        print(
            "使用监督式二分类损失函数 BCEWithLogitsLoss"
            f"（类别平衡 pos_weight=n_neg/n_pos={pos_weight:.4f}）"
        )
        return SupervisedBCEWithLogitsLoss(pos_weight=pos_weight)

    if args.model == "nnpucnn":
        from nnpu_stable_adaptive_pu_loss import (
            StableAdaptivePULoss,
            StableStandardPULoss,
        )

        loss_type = str(getattr(args, "loss_type", "adaptive")).strip().lower()
        if loss_type == "standard":
            loss_func = StableStandardPULoss(
                prior,
                gamma=args.gamma,
                beta=args.beta,
                nnpu=True,
            )
            print(
                "使用 nnPU-CNN 标准 nnPU 损失：固定 gamma、验证不写诊断状态，"
                f"gamma={args.gamma}，beta={args.beta}"
            )
            return loss_func
        if loss_type != "adaptive":
            raise ValueError(
                f"nnPU-CNN 不支持损失类型 {loss_type!r}；请选择 standard 或 adaptive。"
            )

        loss_func = StableAdaptivePULoss(
            prior,
            gamma=args.gamma,
            beta=args.beta,
            adaptive_window=args.adaptive_window,
            nnpu=True,
            adaptive_lambda=getattr(args, "adaptive_lambda", 1.0),
            gamma_min=getattr(args, "adaptive_gamma_min", None),
            gamma_max=getattr(args, "adaptive_gamma_max", None),
            gamma_ema=getattr(args, "adaptive_gamma_ema", 0.8),
            verbose=True,
        )
        print(
            "使用 nnPU-CNN 稳定自适应损失：按 epoch 更新 gamma、验证不写诊断状态，"
            f"初始gamma={args.gamma}，beta={args.beta}，窗口={args.adaptive_window}，"
            f"lambda={getattr(args, 'adaptive_lambda', 1.0)}，"
            f"gamma_ema={getattr(args, 'adaptive_gamma_ema', 0.8)}"
        )
        return loss_func

    if args.model == "rnapucnn":
        from rn_adaptive_pu_loss import RNAdaptivePULoss

        loss_func = RNAdaptivePULoss(

            prior,
            gamma=args.gamma,
            adaptive_window=args.adaptive_window,
            adaptive_lambda=getattr(args, "adaptive_lambda", 1.0),
            gamma_min=getattr(args, "adaptive_gamma_min", None),
            gamma_max=getattr(args, "adaptive_gamma_max", None),
            gamma_ema=getattr(args, "adaptive_gamma_ema", 0.8),
            warmup_epochs=getattr(args, "rn_warmup_epochs", 5),
            rn_fraction=getattr(args, "rn_fraction", 0.25),
            rn_base_weight=getattr(args, "rn_base_weight", 0.10),
            rn_rank_weight=getattr(args, "rn_rank_weight", 0.05),
            rn_rank_margin=getattr(args, "rn_rank_margin", 0.5),
            rn_adaptive_window=getattr(args, "rn_adaptive_window", 10),
            rn_max_weight=getattr(args, "rn_max_weight", 0.30),
        )
        print(
            "使用 RN-aNNPU-CNN 独立损失：现有 AdaptivePULoss + 自适应可靠负样本辅助项，beta 固定为 0"
        )
        return loss_func

    if args.model in {"rnscapucnn", "rnscspucnn"}:
        from rn_conservative_adaptive_pu_loss import RNConservativeAdaptivePULoss

        adaptive_rn = args.model == "rnscapucnn"
        return RNConservativeAdaptivePULoss(
            prior,
            gamma=args.gamma,
            adaptive_window=args.adaptive_window,
            adaptive_lambda=getattr(args, "adaptive_lambda", 1.0),
            gamma_min=getattr(args, "adaptive_gamma_min", None),
            gamma_max=getattr(args, "adaptive_gamma_max", None),
            gamma_ema=getattr(args, "adaptive_gamma_ema", 0.8),
            warmup_epochs=getattr(args, "rn_warmup_epochs", 5),
            rn_fraction=getattr(args, "rn_fraction", 0.20),
            rn_base_weight=getattr(args, "rn_base_weight", 0.10),
            rn_rank_weight=getattr(args, "rn_rank_weight", 0.05),
            rn_rank_margin=getattr(args, "rn_rank_margin", 0.5),
            rn_max_weight=getattr(args, "rn_max_weight", 0.30),
            teacher_momentum=getattr(args, "rn_teacher_momentum", 0.99),
            min_confidence=getattr(args, "rn_min_confidence", 0.55),
            quality_momentum=getattr(args, "rn_quality_momentum", 0.90),
            rn_need_target=getattr(args, "rn_need_target", 0.20),
            rn_min_weight_ratio=getattr(args, "rn_min_weight_ratio", 0.50),
            rn_max_weight_ratio=getattr(args, "rn_max_weight_ratio", 1.25),
            adaptive_rn=adaptive_rn,
        )

    if args.model in {"rncapucnn", "rncpucnn", "rnfcapucnn", "rnfcspucnn", "rngapucnn", "rngspucnn", "rnlrapucnn", "rnlrspucnn"}:
        from rn_confidence_adaptive_pu_loss import RNConfidenceAdaptivePULoss

        adaptive_rn = args.model in {"rncapucnn", "rnfcapucnn", "rngapucnn", "rnlrapucnn"}
        loss_func = RNConfidenceAdaptivePULoss(
            prior,
            gamma=args.gamma,
            adaptive_window=args.adaptive_window,
            adaptive_lambda=getattr(args, "adaptive_lambda", 1.0),
            gamma_min=getattr(args, "adaptive_gamma_min", None),
            gamma_max=getattr(args, "adaptive_gamma_max", None),
            gamma_ema=getattr(args, "adaptive_gamma_ema", 0.8),
            warmup_epochs=getattr(args, "rn_warmup_epochs", 5),
            rn_fraction=getattr(args, "rn_fraction", 0.20),
            rn_base_weight=getattr(args, "rn_base_weight", 0.10),
            rn_rank_weight=getattr(args, "rn_rank_weight", 0.05),
            rn_rank_margin=getattr(args, "rn_rank_margin", 0.5),
            rn_max_weight=getattr(args, "rn_max_weight", 0.30),
            teacher_momentum=getattr(args, "rn_teacher_momentum", 0.99),
            min_confidence=getattr(args, "rn_min_confidence", 0.55),
            quality_momentum=getattr(args, "rn_quality_momentum", 0.90),
            adaptive_rn=adaptive_rn,
        )
        mode_label = "置信度自适应" if adaptive_rn else "固定权重对照"
        print(
            f"使用 {mode_label} RN nnPU 独立损失：现有 AdaptivePULoss + "
            "EMA teacher可靠负样本，beta固定为0"
        )
        return loss_func

    if args.loss_type == 'adaptive':
        loss_func = AdaptivePULoss(
            prior,
            gamma=args.gamma,
            beta=args.beta,
            adaptive_window=args.adaptive_window,
            nnpu=True,
            adaptive_lambda=getattr(args, "adaptive_lambda", 1.0),
            gamma_min=getattr(args, "adaptive_gamma_min", None),
            gamma_max=getattr(args, "adaptive_gamma_max", None),
            gamma_ema=getattr(args, "adaptive_gamma_ema", 0.8),
            verbose=True,
        )
        print(
            f"使用自适应PU损失函数，初始gamma={args.gamma}，beta={args.beta}，"
            f"窗口大小={args.adaptive_window}，lambda={getattr(args, 'adaptive_lambda', 1.0)}，"
            f"gamma_ema={getattr(args, 'adaptive_gamma_ema', 0.8)}，"
            f"gamma_max默认max(gamma,5)"
        )
    else:
        loss_func = PULoss(prior, gamma=args.gamma, beta=args.beta, nnpu=True)
        print(f"使用标准PU损失函数，gamma={args.gamma}，beta={args.beta}")
    return loss_func

def augment_training_tensor(x, noise_std=0.01, seed=42):
    """Apply small feature-space perturbation to training samples only."""
    if x is None:
        return x

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    noise_std = float(noise_std)
    if noise_std <= 0 or x.numel() == 0:
        return x.clone()

    x = x.clone().float()
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))

    feature_std = x.std(dim=0, unbiased=False)
    feature_scale = torch.where(feature_std.abs() < 1e-6, torch.ones_like(feature_std), feature_std)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * noise_std
    return x + noise * feature_scale


def should_drop_last_for_training(num_samples, batch_size):
    """Avoid a final batch of size 1 for BatchNorm1d-based models."""
    batch_size = int(batch_size)
    num_samples = int(num_samples)
    return batch_size > 1 and num_samples > 1 and (num_samples % batch_size) == 1


def _labels_from_tensor_dataset(dataset):
    tensors = getattr(dataset, "tensors", None)
    if not tensors or len(tensors) < 2:
        return None
    labels = tensors[1]
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


def _normalize_positive_sample_weights(dataset):
    """Scale fold-train positive weights to global positive mean one."""
    tensors = getattr(dataset, "tensors", None)
    if not tensors or len(tensors) < 3:
        return dataset
    labels = tensors[1].view(-1)
    weights = tensors[2].view(-1).clone().to(dtype=torch.float32)
    positive = labels == 1
    if not positive.any() or len(weights) != len(labels):
        return dataset
    positive_weights = weights[positive]
    valid = torch.isfinite(positive_weights) & (positive_weights >= 0)
    if not valid.all():
        positive_weights = torch.where(
            valid,
            positive_weights,
            torch.ones_like(positive_weights),
        )
    mean_weight = positive_weights.mean()
    if not torch.isfinite(mean_weight) or float(mean_weight) <= 0:
        positive_weights = torch.ones_like(positive_weights)
        mean_weight = positive_weights.mean()
    weights[positive] = positive_weights / mean_weight
    new_tensors = list(tensors)
    new_tensors[2] = weights.reshape_as(tensors[2])
    return TensorDataset(*new_tensors)


def create_training_data_loader(
    dataset,
    batch_size,
    model_name,
    seed=0,
    positives_per_batch=None,
    unlabeled_coverage=None,
):
    """Create model-specific training loaders while preserving legacy defaults."""
    model_key = str(model_name or "").strip().lower()
    if model_key in {"rnfcapucnn", "rnfcspucnn"}:
        from rn_full_coverage_batch_sampler import (
            RNFullCoveragePositiveUnlabeledBatchSampler,
        )

        labels = _labels_from_tensor_dataset(dataset)
        if labels is None:
            raise ValueError("RN full-coverage training requires labels in a TensorDataset.")
        batch_sampler = RNFullCoveragePositiveUnlabeledBatchSampler(
            labels,
            batch_size=batch_size,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler)
    if model_key == "nnpucnn":
        from pu_batch_sampler import PositiveUnlabeledBatchSampler

        dataset = _normalize_positive_sample_weights(dataset)
        labels = _labels_from_tensor_dataset(dataset)
        if labels is None:
            raise ValueError(
                "nnpucnn training requires labels in a TensorDataset "
                "(positive-guaranteed PU batches)."
            )
        positive_count = 4 if positives_per_batch is None else int(positives_per_batch)
        coverage = 0.5 if unlabeled_coverage is None else float(unlabeled_coverage)
        batch_sampler = PositiveUnlabeledBatchSampler(
            labels,
            batch_size=batch_size,
            seed=seed,
            positives_per_batch=positive_count,
            unlabeled_coverage=coverage,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler)
    if model_key in POSITIVE_BATCH_PU_MODELS:
        from pu_batch_sampler import RNPositiveUnlabeledBatchSampler

        labels = _labels_from_tensor_dataset(dataset)
        if labels is None:
            raise ValueError(
                f"{model_key} training requires labels in a TensorDataset "
                "(positive-guaranteed PU batches)."
            )
        batch_sampler = RNPositiveUnlabeledBatchSampler(
            labels,
            batch_size=batch_size,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=should_drop_last_for_training(len(dataset), batch_size),
    )


def training_loader_diagnostics(data_loader):
    sampler = getattr(data_loader, "batch_sampler", None)
    diagnostics = getattr(sampler, "diagnostics", None)
    return diagnostics() if callable(diagnostics) else None


def save_loss_diagnostics(loss_func, output_path, context=None):
    saver = getattr(loss_func, "save_diagnostics", None)
    if callable(saver):
        return saver(output_path, context=context)
    return None


def create_data_loaders(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    *,
    augmentation_enabled=False,
    augmentation_noise_std=0.01,
    sample_weights=None,
    model_name=None,
    random_seed=0,
    positives_per_batch=None,
    unlabeled_coverage=None,
):
    """创建数据加载器。允许空测试集（全矿点制图 / all_minerals）。"""
    if augmentation_enabled:
        X_train = augment_training_tensor(X_train, noise_std=augmentation_noise_std)
    if sample_weights is None:
        train_dataset = TensorDataset(X_train, y_train)
    else:
        weights = torch.as_tensor(sample_weights, dtype=torch.float32)
        if len(weights) != len(y_train):
            weights = torch.ones(len(y_train), dtype=torch.float32)
        train_dataset = TensorDataset(X_train, y_train, weights)

    # all_minerals / 最终制图：可能无外部测试张量
    def _is_empty_tensor(x):
        if x is None:
            return True
        try:
            return int(len(x)) == 0
        except Exception:
            return True

    if _is_empty_tensor(X_test) or _is_empty_tensor(y_test):
        if X_train is not None and hasattr(X_train, "shape") and len(X_train.shape) >= 1:
            empty_x_shape = (0,) + tuple(X_train.shape[1:])
            X_test = torch.zeros(empty_x_shape, dtype=X_train.dtype if hasattr(X_train, "dtype") else torch.float32)
        else:
            X_test = torch.zeros((0,), dtype=torch.float32)
        y_dtype = y_train.dtype if hasattr(y_train, "dtype") else torch.long
        y_test = torch.zeros((0,), dtype=y_dtype)
        test_dataset = TensorDataset(X_test, y_test)
    else:
        test_dataset = TensorDataset(X_test, y_test)

    train_loader = create_training_data_loader(
        train_dataset,
        batch_size,
        model_name,
        seed=random_seed,
        positives_per_batch=positives_per_batch,
        unlabeled_coverage=unlabeled_coverage,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=(len(test_dataset) > 0))
    
    return train_dataset, train_loader, test_dataset, test_loader
