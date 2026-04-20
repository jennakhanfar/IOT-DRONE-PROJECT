from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DroneFaceDataset
from model import ArcFaceLoss, create_edge_backbone


@dataclass
class DistillationConfig:
    teacher_model: str = "efficientnet_b0"
    student_model: str = "mobilenet_v3_small"
    temperature: float = 4.0
    alpha: float = 0.7
    learning_rate: float = 1e-4
    epochs: int = 5
    batch_size: int = 16
    embedding_dim: int = 256
    pretrained: bool = True
    device: str = "cpu"


class EmbeddingDistillationLoss(torch.nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_embeddings / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_embeddings / self.temperature, dim=1)
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        ce_loss = F.cross_entropy(student_logits, labels)
        feature_loss = 1.0 - F.cosine_similarity(student_embeddings, teacher_embeddings).mean()
        return self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss + 0.25 * feature_loss


def build_teacher_student_models(config: DistillationConfig, num_classes: int):
    teacher, teacher_dim = create_edge_backbone(
        model_name=config.teacher_model,
        pretrained=config.pretrained,
        embedding_dim=config.embedding_dim,
    )
    student, student_dim = create_edge_backbone(
        model_name=config.student_model,
        pretrained=config.pretrained,
        embedding_dim=config.embedding_dim,
    )
    teacher_classifier = ArcFaceLoss(teacher_dim, num_classes).to(config.device)
    student_classifier = ArcFaceLoss(student_dim, num_classes).to(config.device)
    return (
        teacher.to(config.device),
        student.to(config.device),
        teacher_classifier,
        student_classifier,
    )


def train_distilled_student(
    train_root: str,
    output_dir: str,
    config: DistillationConfig,
) -> Dict[str, object]:
    device = torch.device(config.device)
    dataset = DroneFaceDataset(train_root, augment=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    num_classes = len(dataset.classes)

    teacher, student, teacher_classifier, student_classifier = build_teacher_student_models(
        config,
        num_classes,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(student_classifier.parameters()),
        lr=config.learning_rate,
    )
    criterion = EmbeddingDistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
    )

    history = []
    for epoch in range(config.epochs):
        student.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_embeddings = teacher(images)
            student_embeddings = student(images)
            student_logits = student_classifier(student_embeddings, labels)
            loss = criterion(
                student_embeddings,
                teacher_embeddings,
                student_logits,
                labels,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        history.append({"epoch": epoch + 1, "loss": avg_loss})

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_model_state": student.state_dict(),
            "student_classifier_state": student_classifier.state_dict(),
            "config": asdict(config),
            "history": history,
            "classes": dataset.classes,
        },
        output_path / "distilled_student.pth",
    )

    return {
        "config": asdict(config),
        "epochs_completed": config.epochs,
        "final_loss": history[-1]["loss"] if history else None,
        "checkpoint_path": str(output_path / "distilled_student.pth"),
        "history": history,
    }


def export_distillation_plan(output_path: str, config: Optional[DistillationConfig] = None) -> None:
    config = config or DistillationConfig()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(
            [
                "Distillation plan:",
                f"- Teacher model: {config.teacher_model}",
                f"- Student model: {config.student_model}",
                f"- Temperature: {config.temperature}",
                f"- Alpha: {config.alpha}",
                f"- Epochs: {config.epochs}",
                "- Loss: KL distillation + CE + embedding cosine alignment",
            ]
        ),
        encoding="utf-8",
    )
