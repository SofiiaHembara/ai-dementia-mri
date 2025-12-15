# src/models/train_2d_dino_patient.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


# ---------- 1. Аргументи ----------

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--index_csv", type=str, required=True,
                   help="CSV з колонками: subject_id, slice_path, label, split")
    p.add_argument("--slices_root", type=str, default="data",
                   help="Корінь для slice_path (наприклад data або data/processed)")

    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--freeze_epochs", type=int, default=20)
    p.add_argument("--unfreeze_top_k", type=int, default=4)

    p.add_argument("--batch", type=int, default=4, help="кількість пацієнтів у батчі")
    p.add_argument("--size", type=int, default=224)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ft_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--monitor", type=str, default="patient_auc",
                   help="яку метрику моніторити для early stopping")
    p.add_argument("--min_delta", type=float, default=1e-4)

    p.add_argument("--backbone", type=str, default="vit_base_patch16_224.dino")
    p.add_argument("--max_slices_per_patient", type=int, default=48,
                   help="максимум зрізів на пацієнта за одну ітерацію (random subset)")
    p.add_argument("--topk_frac", type=float, default=0.3,
                   help="для top-k агрегації при оцінці (0.3 -> top 30% slice-logits)")

    return p.parse_args()


# ---------- 2. Утиліти препроцесингу ----------

def load_gray_resize(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32)

    m, s = arr.mean(), arr.std() + 1e-6
    arr = (arr - m) / s
    # (H, W) -> (3, H, W)
    arr = np.stack([arr, arr, arr], axis=0)
    return arr  # (3, H, W)


# ---------- 3. Dataset: 1 елемент = 1 пацієнт ----------

class Patient2DDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        slices_root: str | Path = "data",
        max_slices_per_patient: int = 48,
        size: int = 224,  # 👈 ДОДАЛИ
    ):
        self.csv_path = Path(csv_path)
        self.slices_root = Path(slices_root)
        self.split = split
        self.max_slices = max_slices_per_patient
        self.size = size  # 👈 ЗБЕРЕГЛИ РОЗМІР

        df = pd.read_csv(self.csv_path)

        # нормалізуємо split
        df["split"] = df["split"].str.lower().str.strip()
        df = df[df["split"] == split].reset_index(drop=True)

        df["label"] = df["label"].map({"non_demented": 0, "dementia": 1})
        if df["label"].isna().any():
            raise ValueError("Є невідомі label'и в CSV. Очікував тільки 'non_demented'/'dementia'.")

        groups = []
        for subj_id, g in df.groupby("subject_id"):
            slice_paths = g["slice_path"].tolist()
            label = int(g["label"].iloc[0])
            groups.append((subj_id, slice_paths, label))

        self.groups: List[Tuple[str, List[str], int]] = groups
        print(f"[{split}] patients: {len(self.groups)}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int):
        subj_id, slice_paths, label = self.groups[idx]

        if len(slice_paths) > self.max_slices:
            slice_paths = np.random.choice(slice_paths, self.max_slices, replace=False).tolist()

        imgs = []
        for rel_path in slice_paths:
            p = Path(rel_path)

            # 1) якщо шлях вже абсолютний і файл існує — використовуємо як є
            if p.is_absolute() and p.exists():
                final_p = p
            else:
                # 2) якщо він відносний, спочатку пробуємо як є (від кореня проєкту)
                if p.exists():
                    final_p = p
                else:
                    # 3) інакше пробуємо приклеїти slices_root
                    candidate = self.slices_root / p
                    if candidate.exists():
                        final_p = candidate
                    else:
                        raise FileNotFoundError(
                            f"Не можу знайти файл ні за '{p}', ні за '{candidate}'"
                        )

            arr = load_gray_resize(final_p, size=self.size)
            imgs.append(arr)

        imgs = np.stack(imgs, axis=0)  # (N_slices, 3, H, W)
        imgs_t = torch.from_numpy(imgs).float()
        label_t = torch.tensor(label, dtype=torch.float32)

        return imgs_t, label_t, subj_id


# ---------- 4. Collate: кілька пацієнтів → один батч ----------

def patient_collate(batch):
    """
    batch: List[ (imgs: (Ni,3,H,W), label: scalar, subj_id: str) ]

    Повертаємо:
    - all_imgs: (total_slices, 3, H, W)
    - labels: (num_patients,)
    - idx_slices: List[(start, end)] для кожного пацієнта
    - subj_ids: List[str]
    """
    all_imgs = []
    labels = []
    idx_slices: List[Tuple[int, int]] = []
    subj_ids = []

    start = 0
    for imgs, label, subj_id in batch:
        n = imgs.shape[0]
        all_imgs.append(imgs)
        labels.append(label)
        idx_slices.append((start, start + n))
        subj_ids.append(subj_id)
        start += n

    all_imgs = torch.cat(all_imgs, dim=0)  # (total_slices, 3, H, W)
    labels = torch.stack(labels, dim=0)    # (num_patients,)

    return all_imgs, labels, idx_slices, subj_ids


# ---------- 5. Модель DINO + classifier ----------

class DinoClassifier(nn.Module):
    def __init__(self, backbone_name: str = "vit_base_patch16_224.dino"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,   # отримаємо фічі
            in_chans=3,
        )
        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, 1)  # 1 logit (dementia)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)      # (B, D)
        logits = self.head(feats)     # (B, 1)
        return logits


# ---------- 6. Агрегація slice → patient ----------

def aggregate_logits_slice_to_patient(
    slice_logits: torch.Tensor,
    idx_slices: List[Tuple[int, int]],
    mode: str = "mean",
    topk_frac: float = 0.3,
) -> torch.Tensor:
    """
    slice_logits: (total_slices,)  (ще до sigmoid)
    Повертає patient_logits: (num_patients,)
    """
    patient_logits = []
    for (start, end) in idx_slices:
        sl = slice_logits[start:end]  # (Ni,)
        if mode == "topk":
            k = max(1, int(len(sl) * topk_frac))
            topk_vals, _ = torch.topk(sl, k=k)
            patient_logits.append(topk_vals.mean())
        else:  # "mean"
            patient_logits.append(sl.mean())
    return torch.stack(patient_logits, dim=0)


# ---------- 7. Оцінка на split (patient-level) ----------

@torch.no_grad()
def evaluate_on_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    agg_mode: str = "mean",
    topk_frac: float = 0.3,
):
    model.eval()
    all_labels = []
    all_probs = []

    for all_imgs, labels, idx_slices, _ in tqdm(loader, desc="Eval", leave=False):
        all_imgs = all_imgs.to(device)
        labels = labels.to(device)

        logits_slice = model(all_imgs).squeeze(1)  # (total_slices,)
        patient_logits = aggregate_logits_slice_to_patient(
            logits_slice, idx_slices, mode=agg_mode, topk_frac=topk_frac
        )
        probs = torch.sigmoid(patient_logits)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)

    # поріг 0.5 для репорта
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)

    return {"auc": auc, "f1": f1, "bal_acc": ba}


# ---------- 8. Тренування ----------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print("Device:", device)

    # --- дані ---
    global_train_ds = Patient2DDataset(
        csv_path=args.index_csv,
        split="train",
        slices_root=args.slices_root,
        max_slices_per_patient=args.max_slices_per_patient,
        size=args.size,
    )
    val_ds = Patient2DDataset(
        csv_path=args.index_csv,
        split="val",
        slices_root=args.slices_root,
        max_slices_per_patient=args.max_slices_per_patient,
        size=args.size,
    )
    test_ds = Patient2DDataset(
        csv_path=args.index_csv,
        split="test",
        slices_root=args.slices_root,
        max_slices_per_patient=args.max_slices_per_patient,
        size=args.size,
    )

    train_dl = DataLoader(
        global_train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=patient_collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=patient_collate,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=patient_collate,
    )

    # --- модель ---
    model = DinoClassifier(backbone_name=args.backbone).to(device)

    # заморожуємо бекбон спочатку
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    base_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": base_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_metric = -1e9
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for step, (all_imgs, labels, idx_slices, _) in enumerate(pbar):
            all_imgs = all_imgs.to(device)
            labels = labels.to(device)

            logits_slice = model(all_imgs).squeeze(1)  # (total_slices,)
            patient_logits = aggregate_logits_slice_to_patient(
                logits_slice, idx_slices, mode="mean"
            )
            loss = criterion(patient_logits, labels)

            loss.backward()
            if (step + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = labels.shape[0]
            total_loss += loss.item() * bs
            n_samples += bs
            pbar.set_postfix({"loss": total_loss / max(1, n_samples)})

        train_loss = total_loss / max(1, n_samples)

        # --- розморожуємо топ-k блоків після freeze_epochs ---
        if epoch == args.freeze_epochs + 1:
            # простий варіант: розморозити весь backbone
            for p in model.backbone.parameters():
                p.requires_grad = True
            # окремий оптимайзер для fine-tune
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": args.ft_lr},
                    {"params": model.head.parameters(), "lr": args.ft_lr},
                ],
                weight_decay=args.weight_decay,
            )
            print("[INFO] Unfroze backbone for fine-tuning.")

        # --- валідація (patient-level) ---
        val_metrics = evaluate_on_split(
            model, val_dl, device,
            agg_mode="mean", topk_frac=args.topk_frac
        )
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"| VAL patient: AUC={val_metrics['auc']:.3f} "
            f"F1={val_metrics['f1']:.3f} BA={val_metrics['bal_acc']:.3f}"
        )

        monitor_value = val_metrics["auc"]  # можна замінити на іншу

        if monitor_value > best_metric + args.min_delta:
            best_metric = monitor_value
            best_state = model.state_dict()
            epochs_no_improve = 0
            out_path = Path(args.out_dir) / "best_2d_dino_patient.pt"
            torch.save(best_state, out_path)
            print(f"  ↳ ✅ saved best: {out_path} (AUC={best_metric:.3f})")
        else:
            epochs_no_improve += 1
            print(f"  ↳ no improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.patience:
            print("⏹ Early stopping.")
            break

    # --- Тест на кращому чекпоінті ---
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_on_split(
        model, test_dl, device,
        agg_mode="mean", topk_frac=args.topk_frac
    )
    print("TEST patient-level:", test_metrics)


if __name__ == "__main__":
    args = get_args()
    main(args)