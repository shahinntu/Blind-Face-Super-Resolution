import os
import time
import logging
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from src import (
    set_logger,
    clear_handlers,
    save_dict_to_json,
    RunningAverageDict,
    save_checkpoint,
)


class Trainer:
    def __init__(
        self,
        model,
        accelerator,
        config,
        model_log_dir,
        metrics=None,
        objective=None,
    ):
        self._model = model
        self._accelerator = accelerator
        self._config = config
        self._model_log_dir = model_log_dir
        self._metrics = metrics if metrics else {}
        self._objective = objective if objective else "loss"

    @classmethod
    def for_evaluate(cls, model, config, device):
        optimizer = None
        lr_scheduler = None
        model_log_dir = None

        return cls(model, optimizer, lr_scheduler, config, model_log_dir, device)

    def train(self, train_dataloader, val_dataloader=None):
        if self._accelerator.is_main_process:
            log_dir = self._create_log_dirs()
            train_log_path = os.path.join(log_dir, "train_logs", "train.log")
            train_logger = set_logger(train_log_path)
            self._config.save(os.path.join(log_dir, "configs/config.json"))

            summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb_logs"))

            best_obj = float("inf") if self._objective == "loss" else 0

        for epoch in range(self._config.TRAINING.EPOCHS):
            if self._accelerator.is_main_process:
                logging.info(f"Epoch {epoch+1}/{self._config.TRAINING.EPOCHS}")

            train_metrics_avg_dict = self._train_epoch(train_dataloader)

            val_metrics_avg_dict = (
                self.validate(val_dataloader) if val_dataloader else None
            )

            if self._accelerator.is_main_process:
                best_obj = self._log_model_state(
                    epoch,
                    log_dir,
                    val_metrics_avg_dict if val_dataloader else train_metrics_avg_dict,
                    best_obj,
                )
                self._write_tb_logs(
                    summary_writer, epoch, train_metrics_avg_dict, val_metrics_avg_dict
                )

        if self._accelerator.is_main_process:
            summary_writer.close()
            clear_handlers(train_logger)

    @torch.no_grad()
    def validate(self, val_dataloader):
        self._model.eval()

        val_metrics_avg_dict_obj = RunningAverageDict()

        with tqdm(
            total=len(val_dataloader),
            desc="Validating",
            disable=not self._accelerator.is_local_main_process,
        ) as t:
            for val_batch in val_dataloader:
                lq_batch, gt_batch = val_batch
                loss, gt_pred = self._model.val_step(lq_batch, gt_batch)

                val_metrics_dict = {"loss": loss} | {
                    k: self._metrics[k](gt_pred, gt_batch) for k in self._metrics
                }

                val_metrics_avg_dict_obj.update(val_metrics_dict)

                t.set_postfix(metrics=self._format_metrics(val_metrics_avg_dict_obj()))
                t.update()

        val_metrics_avg_dict = self._gather_and_avg_dict(val_metrics_avg_dict_obj)

        if self._accelerator.is_main_process:
            logging.info(
                f"- Validation metrics: {self._format_metrics(val_metrics_avg_dict)}"
            )

        return val_metrics_avg_dict

    def _train_epoch(self, train_dataloader):
        self._model.train()

        train_metrics_avg_dict_obj = RunningAverageDict()

        with tqdm(
            total=len(train_dataloader),
            desc="Training",
            disable=not self._accelerator.is_local_main_process,
        ) as t:
            for train_batch in train_dataloader:
                train_metrics_dict = self._train_step(train_batch)

                train_metrics_avg_dict_obj.update(train_metrics_dict)

                t.set_postfix(
                    metrics=self._format_metrics(train_metrics_avg_dict_obj())
                )
                t.update()

        train_metrics_avg_dict = self._gather_and_avg_dict(train_metrics_avg_dict_obj)

        if self._accelerator.is_main_process:
            logging.info(
                f"- Train metrics: {self._format_metrics(train_metrics_avg_dict)}"
            )

        return train_metrics_avg_dict

    def _train_step(self, train_batch):
        lq_batch, gt_batch, gt_usm_batch = train_batch
        loss, gt_pred = self._model.train_step(
            lq_batch,
            gt_batch,
            gt_usm_batch,
            max_grad_norm=self._config.ACCELERATOR.MAX_GRAD_NORM,
        )

        train_metrics_dict = (
            {"loss": loss}
            | {k: self._metrics[k](gt_pred, gt_batch) for k in self._metrics}
            | {"lr:c": self._model.get_last_learning_rate()}
        )

        return train_metrics_dict

    def _create_log_dirs(self):
        log_dir = os.path.join(
            self._model_log_dir,
            time.strftime("train_%y%m%d%H%M%S", time.localtime(time.time())),
        )
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, "state"))
        os.mkdir(os.path.join(log_dir, "tb_logs"))
        os.mkdir(os.path.join(log_dir, "configs"))
        os.mkdir(os.path.join(log_dir, "train_logs"))

        return log_dir

    def _log_model_state(self, epoch, log_dir, metrics_avg_dict, best_obj):
        if epoch > 200:
            is_best = (
                metrics_avg_dict[self._objective] < best_obj
                if self._objective == "loss"
                else metrics_avg_dict[self._objective] > best_obj
            )
        else:
            is_best = False

        model_states_dict = self._model.get_model_states_dict()
        for model_name, state_dict in model_states_dict.items():
            save_checkpoint(
                {
                    "state_dict": state_dict,
                },
                is_best=is_best,
                checkpoint=os.path.join(log_dir, "state"),
                network_type=model_name,
            )
        last_json_path = os.path.join(log_dir, "state", "last_metrics.json")
        save_dict_to_json(metrics_avg_dict, last_json_path)

        if is_best:
            logging.info("- Found new best objective")
            best_obj = metrics_avg_dict[self._objective]

            best_json_path = os.path.join(log_dir, "state", "best_metrics.json")
            save_dict_to_json(metrics_avg_dict, best_json_path)

        return best_obj

    def _write_tb_logs(self, writer, step, train_metrics, val_metrics):
        for key in train_metrics:
            writer.add_scalar(f"{key}/train", train_metrics[key], step)
        if val_metrics:
            for key in val_metrics:
                writer.add_scalar(f"{key}/validation", val_metrics[key], step)

    def _format_metrics(self, metrics_avg_dict):
        def format_value(v):
            if isinstance(v, float):
                if abs(v) < 1e-3 or abs(v) > 1e3:
                    return f"{v:.2e}"
                else:
                    return f"{v:.4f}"
            else:
                return str(v)

        metrics_string = "; ".join(
            f"{k}: {format_value(v)}" for k, v in metrics_avg_dict.items()
        )
        return metrics_string

    def _gather_and_avg_dict(self, running_average_dict_obj):
        keys, values, steps = running_average_dict_obj.serialize()
        gathered_values = self._accelerator.gather(values.to(self._accelerator.device))
        gathered_steps = self._accelerator.gather(steps.to(self._accelerator.device))
        gathered_dict = {}
        for i, key in enumerate(keys):
            if key.endswith(":c"):
                gathered_dict[key] = gathered_values[:, i].mean().item()
            else:
                gathered_dict[key] = (
                    gathered_values[:, i].sum().item() / gathered_steps.sum().item()
                )
        return gathered_dict
