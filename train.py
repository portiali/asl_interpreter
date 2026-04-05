"""
Train the LandmarkTransformer on collected holistic landmark sequences.

Usage:
    python train.py --data_dir data/ --epochs 100 --batch_size 16
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from src.landmarks import HOLISTIC_VEC_SIZE
from src.model import LandmarkTransformer, CLASS_LABELS


class SignLanguageDataset(Dataset):
    """Load .npy landmark sequences from data/{label}/ directories."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: list[int],
        seq_len: int = 30,
        augment: bool = False,
    ):
        self.sequences = sequences
        self.labels = labels
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = self.sequences[idx].copy()

        if self.augment:
            seq = self._apply_augmentation(seq)

        # Pad or crop to seq_len
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        elif len(seq) > self.seq_len:
            seq = seq[: self.seq_len]

        return torch.tensor(seq, dtype=torch.float32), self.labels[idx]

    def _apply_augmentation(self, seq: np.ndarray) -> np.ndarray:
        # Gaussian noise
        if random.random() < 0.5:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)

        # Horizontal flip: negate x coords, swap left/right hands, flip pose
        if random.random() < 0.5:
            seq = seq.copy()
            hand_dims = 63  # 21 landmarks * 3 dims per hand
            left = seq[:, :hand_dims].copy()
            right = seq[:, hand_dims : hand_dims * 2].copy()
            pose = seq[:, hand_dims * 2 :].copy()
            # Negate x coordinates (every 3rd value starting at 0)
            left[:, 0::3] *= -1
            right[:, 0::3] *= -1
            pose[:, 0::3] *= -1
            # Swap left and right hands
            seq[:, :hand_dims] = right
            seq[:, hand_dims : hand_dims * 2] = left
            seq[:, hand_dims * 2 :] = pose

        # Random time stretch
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            new_len = max(1, int(len(seq) * factor))
            indices = np.linspace(0, len(seq) - 1, new_len).astype(int)
            seq = seq[indices]

        return seq


def load_data(
    data_dir: str, label_map: dict[str, int]
) -> tuple[list[np.ndarray], list[int]]:
    """Walk data directory and load all .npy sequences with their labels."""
    sequences = []
    labels = []

    for label_name, label_idx in label_map.items():
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"  Warning: no directory for '{label_name}', skipping")
            continue

        npy_files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        for f in npy_files:
            seq = np.load(os.path.join(label_dir, f))
            sequences.append(seq)
            labels.append(label_idx)

        print(f"  Loaded {len(npy_files)} sequences for '{label_name}'")

    return sequences, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seqs, targets in loader:
        seqs, targets = seqs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for seqs, targets in loader:
        seqs, targets = seqs.to(device), targets.to(device)
        logits = model(seqs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * seqs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += seqs.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL sign classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Where to save model"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build label map from CLASS_LABELS
    label_map = {name: idx for idx, name in CLASS_LABELS.items()}

    print(f"\nLoading data from {args.data_dir}/")
    sequences, labels = load_data(args.data_dir, label_map)

    if len(sequences) == 0:
        print("No data found! Run collect_data.py first.")
        return

    num_classes = len(set(labels))
    print(f"\nTotal: {len(sequences)} sequences, {num_classes} classes")

    # Train/val split
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = SignLanguageDataset(
        train_seqs, train_labels, seq_len=args.seq_len, augment=True
    )
    val_ds = SignLanguageDataset(
        val_seqs, val_labels, seq_len=args.seq_len, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = LandmarkTransformer(
        num_classes=len(CLASS_LABELS),
        input_size=HOLISTIC_VEC_SIZE,
        seq_len=args.seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop with early stopping
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Load best model and print final evaluation
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"\nBest model — val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    print(f"Saved to {checkpoint_path}")

    # Confusion matrix
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs = seqs.to(device)
            preds = model(seqs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    present = sorted(set(all_targets))
    present_names = [CLASS_LABELS[i] for i in present]

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds, labels=present))
    print("\nClassification Report:")
    print(
        classification_report(
            all_targets, all_preds, labels=present, target_names=present_names
        )
    )


if __name__ == "__main__":
    main()
