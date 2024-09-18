import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator

from src import (
    Params,
    create_hf_train_dataset,
    create_hf_val_dataset,
    FFHQDataset,
    CombinedFilter,
    BufferedDataLoader,
    AutoNetwork,
    SRNetWrapper,
    SRGANWrapper,
    psnr,
    Trainer,
)


class MLPipeline:
    def __init__(self, args):
        self._config = Params.from_json_path(args.config_path)
        self._accelerator = Accelerator(
            mixed_precision=self._config.ACCELERATOR.MIXED_PRECISION,
            gradient_accumulation_steps=self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS,
        )
        self._train_dataloader, self._val_dataloader = self._prepare_data(args)
        self._trainer = self._get_trainer(args)

    def run(self):
        self._trainer.train(self._train_dataloader, self._val_dataloader)

    def _prepare_data(self, args):
        hf_train_dataset = create_hf_train_dataset(os.path.join(args.data_dir, "train"))
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    self._config.DATA.DATA_AUGMENTATION.RESIZE_RESOLUTION
                ),
                (
                    transforms.RandomCrop(
                        self._config.DATA.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                    if self._config.DATA.DATA_AUGMENTATION.RANDOM_CROP
                    else transforms.CenterCrop(
                        self._config.DATA.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if self._config.DATA.DATA_AUGMENTATION.RANDOM_HORIZONTAL_FLIP
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
            ]
        )
        train_dataset = FFHQDataset(hf_train_dataset, train_transform, mode="train")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )
        combined_filter = CombinedFilter(
            self._config.DATA.BLUR,
            self._config.DATA.FILTER,
            self._config.MODEL.GEN.UPSCALE,
            self._config.DATA.DATA_AUGMENTATION.TARGET_RESOLUTION,
            self._accelerator.device,
        )
        buffered_dataloader = BufferedDataLoader(
            train_dataloader,
            self._config.TRAINING.BUFFER_TIMES,
            combined_filter,
            self._accelerator,
        )

        hf_val_dataset = create_hf_val_dataset(os.path.join(args.data_dir, "val"))
        val_transform = transforms.Compose([transforms.ToTensor()])
        val_dataset = FFHQDataset(hf_val_dataset, val_transform, mode="val")
        val_dataloader = DataLoader(
            val_dataset, batch_size=self._config.TRAINING.BATCH_SIZE.TEST, shuffle=False
        )
        val_dataloader = self._accelerator.prepare(val_dataloader)

        return buffered_dataloader, val_dataloader

    def _get_trainer(self, args):
        if args.restore_version:
            log_dir = os.path.join(args.model_log_dir, args.restore_version)
            gen_net = AutoNetwork.from_pretrained(log_dir)
            self._config.add("RESTORED_FROM", args.restore_version)
        else:
            gen_net = AutoNetwork.from_config(self._config.MODEL.GEN)
        gen_opt = AdamW(
            gen_net.parameters(),
            lr=self._config.TRAINING.ADAM_OPTIMIZER.LEARNING_RATE,
            betas=(
                self._config.TRAINING.ADAM_OPTIMIZER.BETA1,
                self._config.TRAINING.ADAM_OPTIMIZER.BETA2,
            ),
            weight_decay=self._config.TRAINING.ADAM_OPTIMIZER.WEIGHT_DECAY,
            eps=self._config.TRAINING.ADAM_OPTIMIZER.EPSILON,
        )
        lr_scheduler = get_scheduler(
            self._config.TRAINING.LR_SCHEDULER.TYPE,
            optimizer=gen_opt,
            num_warmup_steps=round(
                self._config.TRAINING.LR_SCHEDULER.WARMUP_STEPS
                * self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS
            ),
            num_training_steps=round(
                self._config.TRAINING.EPOCHS
                * len(self._train_dataloader)
                / self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS
            ),
        )
        if self._config.MODEL.TYPE == "GAN":
            disc_net = AutoNetwork.from_config(self._config.MODEL.DISC)
            disc_opt = AdamW(
                disc_net.parameters(),
                lr=self._config.TRAINING.ADAM_OPTIMIZER.LEARNING_RATE,
                betas=(
                    self._config.TRAINING.ADAM_OPTIMIZER.BETA1,
                    self._config.TRAINING.ADAM_OPTIMIZER.BETA2,
                ),
                weight_decay=self._config.TRAINING.ADAM_OPTIMIZER.WEIGHT_DECAY,
                eps=self._config.TRAINING.ADAM_OPTIMIZER.EPSILON,
            )
            wrapper = SRGANWrapper(
                gen_net,
                disc_net,
                gen_opt,
                disc_opt,
                lr_scheduler,
                self._accelerator,
                self._config.TRAINING.LOSS,
                self._config.TRAINING.EMA_DECAY,
            )
        elif self._config.MODEL.TYPE == "Net":
            wrapper = SRNetWrapper(
                gen_net,
                gen_opt,
                lr_scheduler,
                self._accelerator,
                self._config.TRAINING.LOSS,
                self._config.TRAINING.EMA_DECAY,
            )

        metrics = {"psnr": psnr}
        objective = "psnr"

        trainer = Trainer(
            wrapper,
            self._accelerator,
            self._config,
            args.model_log_dir,
            metrics=metrics,
            objective=objective,
        )

        return trainer
