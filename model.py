from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return F.normalize(x, p=2, dim=1)


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(cosine_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cosine_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = torch.where(one_hot.bool(), torch.cos(theta + self.m), cosine_theta)
        return output * self.s


class EdgeEmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, embedding_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = EmbeddingHead(
            input_dim=feature_dim,
            hidden_dim=max(feature_dim, embedding_dim),
            embedding_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.head(features)


def infer_feature_dim(backbone: nn.Module, input_size: int = 112) -> int:
    backbone.eval()
    with torch.no_grad():
        sample = torch.randn(1, 3, input_size, input_size)
        features = backbone(sample)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
    return int(features.shape[1])


def create_edge_backbone(
    model_name: str = "mobilenet_v3_small",
    pretrained: bool = True,
    embedding_dim: int = 256,
) -> Tuple[nn.Module, int]:
    model_name = model_name.lower()

    if model_name in {"mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"}:
        try:
            import torchvision.models as models
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torchvision is required for the selected edge backbone."
            ) from exc

        if model_name == "mobilenet_v3_small":
            weights = (
                models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            backbone = models.mobilenet_v3_small(weights=weights)
            backbone.classifier = nn.Identity()
        elif model_name == "mobilenet_v3_large":
            weights = (
                models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            )
            backbone = models.mobilenet_v3_large(weights=weights)
            backbone.classifier = nn.Identity()
        else:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            backbone.classifier = nn.Identity()

        feature_dim = infer_feature_dim(backbone)
        return EdgeEmbeddingModel(backbone, feature_dim, embedding_dim), embedding_dim

    try:
        import timm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "timm is required for model names outside the torchvision presets."
        ) from exc

    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    feature_dim = int(getattr(backbone, "num_features", infer_feature_dim(backbone)))
    return EdgeEmbeddingModel(backbone, feature_dim, embedding_dim), embedding_dim


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(parameter.numel() for parameter in params)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    if torch.backends.quantized.engine == "none":
        supported = torch.backends.quantized.supported_engines
        if "qnnpack" in supported:
            torch.backends.quantized.engine = "qnnpack"
        elif supported:
            torch.backends.quantized.engine = supported[0]
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def apply_global_pruning(model: nn.Module, amount: float = 0.2) -> nn.Module:
    import torch.nn.utils.prune as prune

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        return model

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")
    return model


# Backward-compatible alias for the existing notebook imports.
Embeddinghead = EmbeddingHead
