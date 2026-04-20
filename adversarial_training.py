"""
adversarial_training.py
-----------------------
Adversarial training for mobilenet_v3_small on VGG Face2 (or DroneFaceDataset).

Implements FGSM and PGD attacks during training to make the model robust against
adversarial perturbations — pixel-level noise designed to fool face recognition.
This is the security hardening step before compression (quantization).

Order of operations:
    1. Shamma trains mobilenet_v3_small on VGG Face2 with k-fold  (her script)
    2. This script fine-tunes with adversarial examples              (this file)
    3. Then compress with quantization                                (edge_tools)

Uses only torch (no external adversarial libraries) for Python 3.6.7 compat.

Usage:
    python adversarial_training.py --data-root /path/to/vggface2 --epochs 10
    python adversarial_training.py --data-root /path/to/vggface2 --checkpoint model.pth
    python adversarial_training.py --data-root /path/to/vggface2 --attack pgd --eps 0.03
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ArcFaceLoss, create_edge_backbone

# Try project dataset first, fall back to torchvision ImageFolder
try:
    from dataset import DroneFaceDataset as FaceDataset
    _USING_DRONE_DATASET = True
except ImportError:
    from torchvision.datasets import ImageFolder as FaceDataset
    _USING_DRONE_DATASET = False


# ── Attack implementations (pure torch, no external libs) ────────────────────

def fgsm_attack(model, images, labels, criterion, eps=0.03):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    Single-step attack: perturb each pixel by eps in the direction of the
    loss gradient. Fast but less powerful than PGD.
    """
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = criterion(outputs, labels)
    loss.backward()

    # Perturb in the sign direction of the gradient
    perturbation = eps * images_adv.grad.sign()
    images_adv = (images + perturbation).clamp(0.0, 1.0)
    return images_adv.detach()


def pgd_attack(model, images, labels, criterion, eps=0.03, alpha=0.007,
               steps=7):
    """
    Projected Gradient Descent (Madry et al., 2017).
    Multi-step iterative attack — stronger than FGSM. Each step takes a small
    FGSM-like step (size alpha), then clips back into the eps-ball around the
    original image.
    """
    images_orig = images.clone().detach()
    images_adv = images.clone().detach()
    # Random start within eps-ball
    images_adv = images_adv + torch.empty_like(images_adv).uniform_(-eps, eps)
    images_adv = images_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        images_adv = images_adv.clone().detach().requires_grad_(True)
        outputs = model(images_adv)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()

        # Small FGSM step
        images_adv = images_adv + alpha * images_adv.grad.sign()
        # Project back into eps-ball
        delta = torch.clamp(images_adv - images_orig, min=-eps, max=eps)
        images_adv = (images_orig + delta).clamp(0.0, 1.0)

    return images_adv.detach()


ATTACKS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


# ── Training loop ────────────────────────────────────────────────────────────

def adversarial_train_epoch(model, classifier, loader, optimizer, device,
                            attack_fn, eps, adv_ratio=0.5):
    """
    One epoch of adversarial training.

    For each batch, we train on a mix of clean and adversarial examples:
      - (1 - adv_ratio) of the batch is clean
      - adv_ratio of the batch is adversarially perturbed

    adv_ratio=0.5 means half clean, half adversarial — this prevents the model
    from forgetting how to handle normal images while learning robustness.
    """
    model.train()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Split batch into clean and adversarial portions
        n_adv = max(1, int(batch_size * adv_ratio))
        n_clean = batch_size - n_adv

        clean_images = images[:n_clean]
        adv_source = images[n_clean:]
        adv_labels = labels[n_clean:]

        # Generate adversarial examples (model must be in eval for gradient)
        model.eval()
        # Use a simple CE proxy for generating attacks
        # (ArcFaceLoss needs labels which complicates the attack loop)
        proxy_criterion = nn.CrossEntropyLoss()

        def _attack_forward(x):
            emb = model(x)
            return classifier(emb, adv_labels)

        # Build a thin wrapper so attacks work with our two-part model
        class _AttackWrapper(nn.Module):
            def __init__(self, backbone, head, lbls):
                super(_AttackWrapper, self).__init__()
                self.backbone = backbone
                self.head = head
                self.lbls = lbls

            def forward(self, x):
                emb = self.backbone(x)
                return self.head(emb, self.lbls)

        attack_model = _AttackWrapper(model, classifier, adv_labels)
        attack_model.eval()

        adv_images = attack_fn(
            attack_model, adv_source, adv_labels, proxy_criterion, eps=eps,
        )

        # Combine clean + adversarial
        model.train()
        classifier.train()
        combined_images = torch.cat([clean_images, adv_images], dim=0)
        combined_labels = torch.cat([labels[:n_clean], adv_labels], dim=0)

        # Forward pass on combined batch
        optimizer.zero_grad()
        embeddings = model(combined_images)
        logits = classifier(embeddings, combined_labels)
        loss = F.cross_entropy(logits, combined_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * combined_images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == combined_labels).sum().item()
        total_samples += combined_images.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def evaluate_robustness(model, classifier, loader, device, attack_fn, eps):
    """
    Evaluate model accuracy on clean images vs adversarial images.
    Returns (clean_acc, adversarial_acc) so you can see the robustness gap.
    """
    model.eval()
    classifier.eval()
    proxy_criterion = nn.CrossEntropyLoss()

    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            emb = model(images)
            logits = classifier(emb, labels)
            clean_correct += (logits.argmax(1) == labels).sum().item()

        # Adversarial accuracy
        class _AttackWrapper(nn.Module):
            def __init__(self, backbone, head, lbls):
                super(_AttackWrapper, self).__init__()
                self.backbone = backbone
                self.head = head
                self.lbls = lbls

            def forward(self, x):
                emb = self.backbone(x)
                return self.head(emb, self.lbls)

        attack_model = _AttackWrapper(model, classifier, labels)
        attack_model.eval()

        adv_images = attack_fn(attack_model, images, labels, proxy_criterion,
                               eps=eps)

        with torch.no_grad():
            adv_emb = model(adv_images)
            adv_logits = classifier(adv_emb, labels)
            adv_correct += (adv_logits.argmax(1) == labels).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / max(1, total)
    adv_acc = adv_correct / max(1, total)
    return clean_acc, adv_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial training for mobilenet_v3_small face recognition"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Path to face dataset (VGG Face2 or DroneFaceDataset root).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to pretrained model checkpoint (.pth) from Shamma's k-fold "
             "training. If not provided, starts from ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--model-name", default="mobilenet_v3_small",
        help="timm backbone name (default: mobilenet_v3_small).",
    )
    parser.add_argument(
        "--attack", choices=list(ATTACKS.keys()), default="pgd",
        help="Attack type for adversarial training (default: pgd).",
    )
    parser.add_argument(
        "--eps", type=float, default=0.03,
        help="Perturbation budget (L-inf norm). 0.03 means ~8/255 pixel change. "
             "Default: 0.03.",
    )
    parser.add_argument(
        "--pgd-steps", type=int, default=7,
        help="Number of PGD steps (ignored for FGSM). Default: 7.",
    )
    parser.add_argument(
        "--adv-ratio", type=float, default=0.5,
        help="Fraction of each batch that is adversarial (default: 0.5).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", default="adversarial_runs",
        help="Directory to save the adversarially-trained checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    attack_fn = ATTACKS[args.attack]

    # If using PGD, wrap with extra kwargs
    if args.attack == "pgd":
        _base_fn = attack_fn
        def attack_fn(model, images, labels, criterion, eps=0.03):
            return _base_fn(model, images, labels, criterion, eps=eps,
                            steps=args.pgd_steps)

    # ── Load dataset ─────────────────────────────────────────────────────
    print("[adv] Loading dataset from %s" % args.data_root)
    if _USING_DRONE_DATASET:
        dataset = FaceDataset(args.data_root, augment=True)
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        dataset = FaceDataset(args.data_root, transform=transform)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    else:
        num_classes = len(dataset.class_to_idx)
    print("[adv] Dataset: %d samples, %d classes" % (len(dataset), num_classes))

    # ── Build model ──────────────────────────────────────────────────────
    model, embed_dim = create_edge_backbone(
        model_name=args.model_name,
        pretrained=True,
        embedding_dim=args.embedding_dim,
    )
    classifier = ArcFaceLoss(embed_dim, num_classes)

    # Load Shamma's pretrained checkpoint if provided
    if args.checkpoint:
        print("[adv] Loading checkpoint: %s" % args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "student_model_state" in ckpt:
            model.load_state_dict(ckpt["student_model_state"])
        else:
            # Assume it's a raw state dict
            model.load_state_dict(ckpt)
        if "classifier_state" in ckpt:
            classifier.load_state_dict(ckpt["classifier_state"])
        print("[adv] Checkpoint loaded.")

    model = model.to(device)
    classifier = classifier.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
    )

    # ── Evaluate robustness BEFORE adversarial training ──────────────────
    print("\n[adv] Evaluating robustness BEFORE adversarial training...")
    clean_acc_before, adv_acc_before = evaluate_robustness(
        model, classifier, loader, device, attack_fn, args.eps,
    )
    print("[adv] Before — clean acc: %.4f | adversarial acc: %.4f"
          % (clean_acc_before, adv_acc_before))

    # ── Adversarial training loop ────────────────────────────────────────
    print("\n[adv] Starting adversarial training (%s, eps=%.4f, %d epochs)"
          % (args.attack.upper(), args.eps, args.epochs))
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, acc = adversarial_train_epoch(
            model, classifier, loader, optimizer, device,
            attack_fn, args.eps, args.adv_ratio,
        )
        elapsed = time.time() - t0
        print("[adv] Epoch %d/%d — loss: %.4f  acc: %.4f  (%.1fs)"
              % (epoch, args.epochs, loss, acc, elapsed))
        history.append({
            "epoch": epoch, "loss": loss, "accuracy": acc,
            "elapsed_s": round(elapsed, 1),
        })

    # ── Evaluate robustness AFTER adversarial training ───────────────────
    print("\n[adv] Evaluating robustness AFTER adversarial training...")
    clean_acc_after, adv_acc_after = evaluate_robustness(
        model, classifier, loader, device, attack_fn, args.eps,
    )
    print("[adv] After  — clean acc: %.4f | adversarial acc: %.4f"
          % (clean_acc_after, adv_acc_after))
    print("[adv] Robustness gain: %.4f → %.4f (+%.4f)"
          % (adv_acc_before, adv_acc_after, adv_acc_after - adv_acc_before))

    # ── Save checkpoint + results ────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "adversarial_mobilenet_v3_small.pth"
    torch.save({
        "model_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "config": {
            "attack": args.attack,
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "adv_ratio": args.adv_ratio,
            "epochs": args.epochs,
            "lr": args.lr,
            "model_name": args.model_name,
        },
        "history": history,
    }, ckpt_path)

    results_path = out_dir / "adversarial_results.json"
    results = {
        "before": {
            "clean_accuracy": clean_acc_before,
            "adversarial_accuracy": adv_acc_before,
        },
        "after": {
            "clean_accuracy": clean_acc_after,
            "adversarial_accuracy": adv_acc_after,
        },
        "robustness_gain": round(adv_acc_after - adv_acc_before, 4),
        "clean_accuracy_change": round(clean_acc_after - clean_acc_before, 4),
        "config": {
            "attack": args.attack,
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "adv_ratio": args.adv_ratio,
            "epochs": args.epochs,
        },
        "history": history,
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n[adv] Saved checkpoint: %s" % ckpt_path)
    print("[adv] Saved results:    %s" % results_path)
    print("\n[adv] Next steps:")
    print("  1. Check that clean accuracy didn't drop too much")
    print("  2. Run edge_tools compression on the adversarial checkpoint")
    print("  3. Re-benchmark under drone constraints")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
