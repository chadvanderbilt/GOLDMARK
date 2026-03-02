from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from goldmark.training.aggregators import create_aggregator
from goldmark.training.datasets import DatasetConfig, SlideLevelDataset, collate_fn
from goldmark.utils.logging import get_logger


@dataclass
class TrainerConfig:
    aggregator: str = "gma"
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    split_column: str = "split"
    train_split_value: str = "train"
    val_split_value: str = "val"
    test_split_value: Optional[str] = "test"
    dropout: bool = True
    device: str = "cuda"
    num_workers: int = 0
    class_weight_positive: Optional[float] = None
    encoder_name: Optional[str] = None
    val_interval: int = 1
    extra_val_epochs: Optional[list] = None


class MILTrainer:
    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_dir: Optional[Path],
        output_dir: Path,
        target_column: str,
        config: TrainerConfig,
        log_level: str = "INFO",
    ) -> None:
        self.manifest = manifest
        self.feature_dir = Path(feature_dir).resolve() if feature_dir else None
        self.output_dir = Path(output_dir)
        self.target_column = target_column
        self.config = config
        self.logger = get_logger(__name__, level=log_level)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Move legacy checkpoints into checkpoint/ if they exist at the split root.
        for pattern in ("checkpoint_*.pt", "checkpoint_*.pth"):
            for path in self.output_dir.glob(pattern):
                if path.is_file():
                    dest = self.checkpoint_dir / path.name
                    if not dest.exists():
                        path.rename(dest)
        self.plot_entries: List[dict] = []
        interval = getattr(config, "val_interval", 1) or 1
        self.val_interval = max(int(interval), 1)
        extra = getattr(config, "extra_val_epochs", None) or []
        self.extra_val_epochs = {int(e) for e in extra}

    def run(self) -> Dict[str, object]:
        config_path = self.output_dir / "trainer_config.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=2))

        train_dataset = SlideLevelDataset(
            self.manifest,
            DatasetConfig(
                feature_dir=self.feature_dir,
                target_column=self.target_column,
                split_column=self.config.split_column,
                subset_value=self.config.train_split_value,
            ),
        )
        val_dataset = SlideLevelDataset(
            self.manifest,
            DatasetConfig(
                feature_dir=self.feature_dir,
                target_column=self.target_column,
                split_column=self.config.split_column,
                subset_value=self.config.val_split_value,
            ),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )
        self._validate_dataset_features(train_dataset, "training")
        self._validate_dataset_features(val_dataset, "validation")

        feature_dim = self._infer_feature_dim(train_dataset)
        num_classes = self._num_classes(train_dataset, val_dataset)
        model = create_aggregator(
            self.config.aggregator,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=self.config.dropout,
        )
        model.to(self.device)

        class_weights = None
        if self.config.class_weight_positive is not None:
            weight = torch.tensor(
                [
                    1.0 - self.config.class_weight_positive,
                    self.config.class_weight_positive,
                ],
                device=self.device,
            )
            class_weights = weight

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

        best_auc = -np.inf
        epochs_no_improve = 0
        history = []
        best_epoch = 0
        best_metrics: Dict[str, float] = {}
        best_val_records: Optional[List[dict]] = None

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            run_validation = self._should_run_validation(epoch)
            val_metrics: Dict[str, float] = {}
            val_records: Optional[List[dict]] = None
            if run_validation:
                val_metrics, val_records = self._evaluate(
                    model,
                    val_loader,
                    stage="val",
                    epoch=epoch,
                    record_outputs=True,
                    capture_outputs=True,
                )
                scheduler.step(val_metrics.get("roc_auc", 0.0))
                self._write_epoch_metrics(epoch, train_loss, val_metrics)
                if val_records:
                    self._write_probability_file(val_records, stage="validation", epoch=epoch)
                    self._write_stage_metrics(val_metrics, stage="validation", epoch=epoch)
                    self._write_confusion_matrix_artifacts(val_records, stage="validation", epoch=epoch)
            else:
                self._write_epoch_metrics(epoch, train_loss, {})

            history.append(
                {
                    "epoch": epoch,
                    "training_loss": train_loss,
                    "macro_auc": val_metrics.get("roc_auc", float("nan")),
                    "macro_ber": val_metrics.get("balanced_error_rate", float("nan")),
                    "macro_fpr": val_metrics.get("fpr", float("nan")),
                    "macro_fnr": val_metrics.get("fnr", float("nan")),
                    "accuracy": val_metrics.get("accuracy", float("nan")),
                    "precision": val_metrics.get("precision", float("nan")),
                    "recall": val_metrics.get("recall", float("nan")),
                    "f1": val_metrics.get("f1", float("nan")),
                }
            )
            if run_validation:
                self.logger.info(
                    "Epoch %d | loss %.4f | val_auc %.4f | acc %.4f",
                    epoch,
                    train_loss,
                    val_metrics.get("roc_auc", float("nan")),
                    val_metrics.get("accuracy", float("nan")),
                )
            else:
                next_eval = self._next_validation_epoch(epoch)
                self.logger.info(
                    "Epoch %d | loss %.4f | validation deferred (next at epoch %d)",
                    epoch,
                    train_loss,
                    next_eval,
                )
            self._write_progress(epoch, status="running")

            if not run_validation:
                continue

            self._save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                num_classes,
                filename=f"checkpoint_epoch_{epoch:03d}.pt",
            )

            current_auc = val_metrics.get("roc_auc", float("-inf"))
            if not np.isfinite(current_auc):
                current_auc = float("-inf")
            should_save = current_auc > best_auc or not np.isfinite(best_auc)

            if should_save:
                best_auc = current_auc
                epochs_no_improve = 0
                best_epoch = epoch
                best_metrics = val_metrics
                best_val_records = list(val_records or [])
                self._save_checkpoint(model, optimizer, epoch, val_metrics, num_classes)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config.patience:
                self.logger.info("Early stopping after %d epochs", epoch)
                break

        self._save_history(history)
        self._mirror_classification_report(stage="validation")
        metrics_summary: Dict[str, Dict[str, float]] = {
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
        }

        test_value = self.config.test_split_value
        if test_value:
            test_dataset = SlideLevelDataset(
                self.manifest,
                DatasetConfig(
                    feature_dir=self.feature_dir,
                    target_column=self.target_column,
                    split_column=self.config.split_column,
                    subset_value=test_value,
                ),
            )
            self._validate_dataset_features(test_dataset, "test")
            if len(test_dataset) > 0:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    collate_fn=collate_fn,
                )
                checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint["model_state"])
                    model.to(self.device)
                test_metrics, test_records = self._evaluate(
                    model,
                    test_loader,
                    stage="test",
                    epoch=best_epoch or epoch,
                    capture_outputs=True,
                )
                metrics_summary["test_metrics"] = test_metrics
                self._write_probability_file(test_records, stage="test", epoch=best_epoch or epoch)
                self._write_stage_metrics(test_metrics, stage="test", epoch=best_epoch or epoch)
                self._write_confusion_matrix_artifacts(test_records, stage="test", epoch=best_epoch or epoch)
                self._write_best_model_metadata(stage="test", epoch=best_epoch or epoch, metrics=test_metrics)
                self._save_metrics({"best_epoch": best_epoch, "best_val": best_metrics, "test": test_metrics})
                self._capture_plot_curves(
                    test_records,
                    stage="test",
                    epoch=best_epoch or epoch,
                )
            else:
                metrics_summary["test_metrics"] = {}
                self._save_metrics({"best_epoch": best_epoch, "best_val": best_metrics, "test": None})
        else:
            metrics_summary["test_metrics"] = None
            self._save_metrics({"best_epoch": best_epoch, "best_val": best_metrics, "test": None})

        if best_val_records:
            selected_epoch = best_epoch or epoch
            self._write_probability_file(best_val_records, stage="validation", epoch=selected_epoch)
            self._write_stage_metrics(best_metrics, stage="validation", epoch=selected_epoch)
            self._write_confusion_matrix_artifacts(best_val_records, stage="validation", epoch=selected_epoch)
            self._write_best_model_metadata(stage="validation", epoch=selected_epoch, metrics=best_metrics)
            self._capture_plot_curves(
                best_val_records,
                stage="validation",
                epoch=selected_epoch,
            )

        final_epoch = best_epoch or epoch
        self._write_progress(final_epoch, status="completed")
        metrics_summary["plot_entries"] = list(self.plot_entries)
        return metrics_summary

    # ------------------------------------------------------------------
    def _train_epoch(self, model, loader, criterion, optimizer) -> float:
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            optimizer.zero_grad()
            _, _, logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
            total_examples += targets.size(0)
        return total_loss / max(total_examples, 1)

    def _evaluate(
        self,
        model,
        loader,
        *,
        stage: str = "val",
        epoch: Optional[int] = None,
        record_outputs: bool = False,
        capture_outputs: bool = False,
    ) -> tuple[Dict[str, float], Optional[List[dict]]]:
        model.eval()
        all_targets: list[int] = []
        all_probs: list[float] = []
        all_preds: list[int] = []
        need_records = record_outputs or capture_outputs
        batch_records: List[dict] = [] if need_records else []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                slide_ids = batch.get("slide_ids") or []
                _, _, logits = model(features)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                target_values = targets.cpu().tolist()
                prob_values = probs.cpu().tolist()
                pred_values = preds.cpu().tolist()
                all_targets.extend(target_values)
                all_probs.extend(prob_values)
                all_preds.extend(pred_values)
                if need_records and slide_ids:
                    for sid, target_val, prob_val, pred_val in zip(slide_ids, target_values, prob_values, pred_values):
                        batch_records.append(
                            {
                                "slide_id": sid,
                                "target": int(target_val),
                                "prob_positive": float(prob_val),
                                "prediction": int(pred_val),
                                "stage": stage,
                                "epoch": epoch,
                            }
                        )

        if not all_targets:
            null_metrics = {"roc_auc": float("nan"), "accuracy": float("nan")}
            return null_metrics, (batch_records if capture_outputs else None)

        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(all_targets, all_probs)
        except ValueError:
            metrics["roc_auc"] = float("nan")
        tn, fp, fn, tp = self._confusion_counts(all_targets, all_preds)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics["fpr"] = fpr
        metrics["fnr"] = fnr
        metrics["balanced_error_rate"] = 0.5 * (fpr + fnr)
        if record_outputs and batch_records:
            self._append_eval_outputs(batch_records, stage=stage, epoch=epoch)
        return metrics, (batch_records if capture_outputs else None)

    def _save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: Dict[str, float],
        num_classes: int,
        filename: str = "checkpoint_best.pt",
    ) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "config": asdict(self.config),
                "target_column": self.target_column,
                "num_classes": num_classes,
            },
            path,
        )
        self.logger.info("Saved checkpoint to %s", path)

    def _write_progress(self, epoch: int, status: str = "running") -> None:
        try:
            progress_path = self.output_dir / "progress.json"
            payload = {
                "target": self.target_column,
                "encoder": self.config.encoder_name,
                "current_epoch": max(int(epoch), 0),
                "total_epochs": int(self.config.epochs),
                "status": status,
                "updated_at": datetime.utcnow().isoformat(),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload, indent=2))
        except Exception:
            # Progress tracking should not break training; swallow errors.
            pass

    def _save_history(self, history) -> None:
        history_path = self.output_dir / "training_history.json"
        history_path.write_text(json.dumps(history, indent=2))
        report_dir = self.output_dir / "classification_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        history_df = pd.DataFrame(history)
        history_df.to_csv(report_dir / "cumulative_results.csv", index=False)

    def _validate_dataset_features(self, dataset, stage: str) -> None:
        degenerate = getattr(dataset, "degenerate_entries", None) or []
        if not degenerate:
            return
        preview = ", ".join(
            f"{entry.get('slide_id') or Path(entry.get('feature_path', '')).stem}"
            for entry in degenerate[:5]
        )
        raise ValueError(
            f"{stage.title()} dataset contains feature tensors where all tile embeddings are identical "
            f"(e.g., {preview}). Re-run feature extraction for these slides before training."
        )

    def _save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        report_dir = self.output_dir / "classification_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "summary.json").write_text(json.dumps(metrics, indent=2))

    def _stage_labels(self, stage: str) -> tuple[str, str]:
        normalized = (stage or "").lower()
        if normalized.startswith("val"):
            return "cross_validation", "validation"
        if normalized.startswith("test"):
            return "test", "test"
        return normalized or "unknown", normalized or "unknown"

    def _stage_dir(self, stage_key: str) -> Path:
        path = self.output_dir / "inference" / stage_key
        path.mkdir(parents=True, exist_ok=True)
        for child in ("confusion_matrix", "metrics", "multiclass_plots", "predictions"):
            (path / child).mkdir(parents=True, exist_ok=True)
        return path

    def _write_stage_metrics(self, metrics: Dict[str, float], stage: str, epoch: Optional[int]) -> None:
        if not metrics:
            return
        stage_key, _ = self._stage_labels(stage)
        stage_dir = self._stage_dir(stage_key)
        metrics_dir = stage_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                continue
        payload["epoch"] = int(epoch or -1)
        payload["updated_at"] = datetime.utcnow().isoformat()
        filename = metrics_dir / (
            f"metrics_epoch_{int(epoch):03d}.json" if epoch is not None else "metrics_summary.json"
        )
        filename.write_text(json.dumps(payload, indent=2))

    def _write_confusion_matrix_artifacts(
        self,
        records: Optional[List[dict]],
        stage: str,
        epoch: Optional[int],
    ) -> None:
        if not records:
            return
        y_true = []
        y_pred = []
        for row in records:
            try:
                y_true.append(int(row.get("target", 0)))
            except (TypeError, ValueError):
                y_true.append(0)
            if "prediction" in row:
                try:
                    y_pred.append(int(row["prediction"]))
                except (TypeError, ValueError):
                    y_pred.append(0)
            else:
                prob = float(row.get("prob_positive", 0.0))
                y_pred.append(1 if prob >= 0.5 else 0)
        if not y_true:
            return
        stage_key, _ = self._stage_labels(stage)
        stage_dir = self._stage_dir(stage_key)
        cm_dir = stage_dir / "confusion_matrix"
        cm_dir.mkdir(parents=True, exist_ok=True)
        try:
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        except Exception:
            return
        payload = {
            "epoch": int(epoch or -1),
            "labels": [0, 1],
            "matrix": matrix,
            "updated_at": datetime.utcnow().isoformat(),
        }
        filename = cm_dir / (
            f"confusion_epoch_{int(epoch):03d}.json" if epoch is not None else "confusion_summary.json"
        )
        filename.write_text(json.dumps(payload, indent=2))

    def _write_best_model_metadata(self, stage: str, epoch: Optional[int], metrics: Dict[str, float]) -> None:
        if epoch is None:
            return
        stage_key, _ = self._stage_labels(stage)
        stage_dir = self._stage_dir(stage_key)
        checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        payload = {
            "best_epoch": int(epoch),
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
            "generated_at": datetime.utcnow().isoformat(),
        }
        (stage_dir / "best_model.json").write_text(json.dumps(payload, indent=2))
        epochs_payload = {
            "selected_epochs": [int(epoch)],
            "total_epochs": int(self.config.epochs),
            "generated_at": datetime.utcnow().isoformat(),
        }
        (stage_dir / "epochs_selected.json").write_text(json.dumps(epochs_payload, indent=2))

    def _should_run_validation(self, epoch: int) -> bool:
        if epoch in self.extra_val_epochs:
            return True
        if self.val_interval <= 1:
            return True
        if epoch >= self.config.epochs:
            return True
        return epoch % self.val_interval == 0

    def _next_validation_epoch(self, epoch: int) -> int:
        if self.val_interval <= 1:
            return min(epoch + 1, self.config.epochs)
        if epoch >= self.config.epochs:
            return self.config.epochs
        if epoch < self.val_interval:
            return min(self.val_interval, self.config.epochs)
        candidate = ((epoch // self.val_interval) + 1) * self.val_interval
        return min(candidate, self.config.epochs)

    def _mirror_classification_report(self, stage: str) -> None:
        source = self.output_dir / "classification_report"
        if not source.exists():
            return
        stage_key, _ = self._stage_labels(stage)
        dest = self._stage_dir(stage_key) / "classification_report"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, dirs_exist_ok=True)

    def _infer_feature_dim(self, dataset: SlideLevelDataset) -> int:
        if len(dataset) == 0:
            raise ValueError("Training dataset is empty")
        sample = dataset[0]["features"]
        return int(sample.shape[1])

    def _num_classes(self, train_dataset: SlideLevelDataset, val_dataset: SlideLevelDataset) -> int:
        targets = []
        for ds in (train_dataset, val_dataset):
            targets.extend(ds.records[self.target_column].dropna().unique().tolist())
        unique_targets = sorted(set(int(t) for t in targets))
        return max(len(unique_targets), 2)

    def _confusion_counts(self, targets, preds) -> tuple[int, int, int, int]:
        tn = fp = fn = tp = 0
        for target, pred in zip(targets, preds):
            if target == 1 and pred == 1:
                tp += 1
            elif target == 1 and pred == 0:
                fn += 1
            elif target == 0 and pred == 1:
                fp += 1
            else:
                tn += 1
        return tn, fp, fn, tp

    def _append_eval_outputs(self, records: List[dict], stage: str, epoch: Optional[int]) -> None:
        if not records:
            return
        report_dir = self.output_dir / "classification_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        outputs_path = report_dir / "validation_outputs.csv"
        df = pd.DataFrame(records)
        columns = [
            "stage",
            "epoch",
            "slide_id",
            "target",
            "prob_positive",
            "prediction",
        ]
        df = df[columns]
        mode = "a" if outputs_path.exists() else "w"
        header = not outputs_path.exists()
        df.to_csv(outputs_path, mode=mode, header=header, index=False)
        if stage.startswith("val") and epoch is not None:
            epoch_dir = report_dir / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            epoch_file = epoch_dir / f"validation_epoch_{int(epoch):03d}_outputs.csv"
            df.to_csv(epoch_file, index=False)

    def _write_epoch_metrics(self, epoch: int, train_loss: float, metrics: Dict[str, float]) -> None:
        report_dir = self.output_dir / "classification_report" / "epochs"
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {"epoch": int(epoch), "training_loss": float(train_loss)}
        for key, value in metrics.items():
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                continue
        df = pd.DataFrame([payload])
        df.to_csv(report_dir / f"validation_epoch_{int(epoch):03d}_metrics.csv", index=False)

    def _write_probability_file(
        self,
        records: Optional[List[dict]],
        stage: str,
        epoch: Optional[int] = None,
    ) -> None:
        if not records:
            return
        df = pd.DataFrame(records)
        if df.empty or "prob_positive" not in df.columns:
            return
        probs = pd.to_numeric(df["prob_positive"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        targets = pd.to_numeric(df.get("target"), errors="coerce").fillna(0).astype(int)
        raw_slide_ids = df.get("slide_id")
        fallback = pd.Series(
            [f"slide_{idx}" for idx in range(len(df))],
            index=df.index,
            dtype="string",
        )
        if raw_slide_ids is None:
            slide_ids = fallback
        else:
            slide_ids = raw_slide_ids.astype("string")
            mask = slide_ids.isna() | slide_ids.str.strip().eq("")
            lower = slide_ids.str.lower()
            mask = mask | lower.isin(["nan", "inf", "-inf"])
            slide_ids = slide_ids.mask(mask, fallback)
            slide_ids = slide_ids.fillna(fallback)
        payload = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "prob_class0": 1.0 - probs,
                "prob_class1": probs,
                "targets": targets,
            }
        )
        stage_key, legacy_stage = self._stage_labels(stage)
        stage_dir = self._stage_dir(stage_key)
        filename_prefix = f"probabilities_{stage_key}"
        if epoch is not None:
            epoch_file = stage_dir / f"{filename_prefix}_epoch_{int(epoch):03d}.csv"
            payload.to_csv(epoch_file, index=False)
            predictions_dir = stage_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(epoch_file, predictions_dir / epoch_file.name)
        aggregate_stage_path = stage_dir / f"{filename_prefix}_set.csv"
        payload.to_csv(aggregate_stage_path, index=False)
        legacy_prefix = f"probabilities_{legacy_stage}"
        legacy_path = self.output_dir / f"{legacy_prefix}_set.csv"
        payload.to_csv(legacy_path, index=False)
        curve_records: List[Dict[str, Any]]
        if records:
            curve_records = records
        else:
            curve_records = [
                {
                    "slide_id": row.get("slide_id"),
                    "target": row.get("targets"),
                    "prob_positive": row.get("prob_class1"),
                    "stage": stage,
                    "epoch": epoch,
                }
                for row in payload.to_dict(orient="records")
            ]
        self._capture_plot_curves(curve_records, stage=stage, epoch=epoch)

    def _capture_plot_curves(
        self,
        records: Optional[List[dict]],
        *,
        stage: str,
        epoch: Optional[int],
    ) -> None:
        curves = self._build_curve_payload(records)
        if not curves:
            return
        stage_key, _ = self._stage_labels(stage)
        entry = {
            "target": self.target_column,
            "encoder": self.config.encoder_name,
            "aggregator": self.config.aggregator,
            "split": self.config.split_column,
            "stage": stage_key,
            "epoch": int(epoch or -1),
            "generated_at": datetime.utcnow().isoformat(),
        }
        entry.update(curves)
        self.plot_entries.append(entry)
        self._write_plot_artifact(stage_key, entry)

    def _write_plot_artifact(self, stage_key: str, entry: Dict[str, Any]) -> None:
        plot_dir = self._stage_dir(stage_key) / "multiclass_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        filename = self._plot_artifact_name(stage_key, entry.get("epoch"))
        payload = {
            "target": entry.get("target"),
            "encoder": entry.get("encoder"),
            "aggregator": entry.get("aggregator"),
            "split": entry.get("split"),
            "stage": stage_key,
            "epoch": entry.get("epoch"),
            "generated_at": entry.get("generated_at"),
            "roc": entry.get("roc"),
            "pr": entry.get("pr"),
            "auc": entry.get("auc"),
        }
        try:
            (plot_dir / filename).write_text(json.dumps(payload, indent=2))
        except OSError:
            pass

    def _plot_artifact_name(self, stage_key: str, epoch: Optional[int]) -> str:
        parts = [
            str(self.target_column or "target"),
            str(self.config.encoder_name or "encoder"),
            stage_key or "stage",
        ]
        if epoch and int(epoch) >= 0:
            parts.append(f"epoch_{int(epoch):03d}")
        base = "_".join(parts)
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", base).strip("_").lower()
        return f"{slug or 'plot'}.json"

    def _build_curve_payload(self, records: Optional[List[dict]]) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        df = pd.DataFrame(records)
        if df.empty or "prob_positive" not in df.columns or "target" not in df.columns:
            return None
        frame = df[["target", "prob_positive"]].dropna()
        if frame.empty:
            return None
        y_true = pd.to_numeric(frame["target"], errors="coerce").dropna()
        y_scores = pd.to_numeric(frame["prob_positive"], errors="coerce").dropna()
        aligned = frame.loc[y_true.index.intersection(y_scores.index)]
        if aligned.empty:
            return None
        labels = aligned["target"].astype(int)
        scores = aligned["prob_positive"].astype(float).clip(0.0, 1.0)
        has_pos = bool((labels == 1).any())
        has_neg = bool((labels == 0).any())
        payload: Dict[str, Any] = {}
        if has_pos and has_neg:
            try:
                fpr, tpr, _ = roc_curve(labels, scores)
                fpr_ds, tpr_ds = self._downsample_pair(fpr, tpr)
                payload["roc"] = {
                    "fpr": fpr_ds,
                    "tpr": tpr_ds,
                }
                payload["auc"] = float(roc_auc_score(labels, scores))
            except ValueError:
                pass
        if has_pos:
            try:
                precision, recall, _ = precision_recall_curve(labels, scores)
                recall_ds, precision_ds = self._downsample_pair(recall, precision)
                payload["pr"] = {
                    "recall": recall_ds,
                    "precision": precision_ds,
                }
                payload["ap"] = float(average_precision_score(labels, scores))
            except ValueError:
                pass
        return payload or None

    def _downsample_pair(self, x_values: Iterable[float], y_values: Iterable[float], max_points: int = 200) -> Tuple[List[float], List[float]]:
        x = np.asarray(list(x_values), dtype=float)
        y = np.asarray(list(y_values), dtype=float)
        if x.size == 0 or x.size != y.size:
            return [], []
        if x.size <= max_points:
            return x.round(6).tolist(), y.round(6).tolist()
        indices = np.linspace(0, x.size - 1, max_points)
        sampled_idx = np.round(indices).astype(int)
        return x[sampled_idx].round(6).tolist(), y[sampled_idx].round(6).tolist()
