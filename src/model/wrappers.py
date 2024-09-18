import copy

import torch
from basicsr.losses.basic_loss import L1Loss, MSELoss, PerceptualLoss
from basicsr.losses.gan_loss import GANLoss


class SRNetWrapper:
    def __init__(
        self, gen_net, gen_opt, lr_scheduler, accelerator, loss_config, ema_decay=0
    ):
        self._gen_net = gen_net
        self._gen_opt = gen_opt
        self._lr_scheduler = lr_scheduler
        self._accelerator = accelerator
        self._loss_config = loss_config
        self._ema_decay = ema_decay

        self._pix_criterion = (
            L1Loss(loss_config.PIXEL_LOSS.WEIGHT).to(accelerator.device)
            if loss_config.PIXEL_LOSS.TYPE == "l1"
            else MSELoss(loss_config.PIXEL_LOSS.WEIGHT).to(accelerator.device)
        )

        self._gen_net, self._gen_opt, self._lr_scheduler = self._accelerator.prepare(
            self._gen_net, self._gen_opt, self._lr_scheduler
        )

        self._initialize_gen_net_ema()

    def train_step(self, *args, max_grad_norm=1):
        with self._accelerator.accumulate(self._gen_net):
            lq, gt, gt_usm = args

            pix_gt = gt
            if self._loss_config.PIXEL_LOSS.GT_USM:
                pix_gt = gt_usm

            self._gen_opt.zero_grad()
            pred_gt = self._gen_net(lq)
            loss = self._pix_criterion(pred_gt, pix_gt)
            self._accelerator.backward(loss)
            if self._accelerator.sync_gradients:
                self._accelerator.clip_grad_norm_(
                    self._gen_net.parameters(), max_grad_norm
                )
            self._gen_opt.step()
            self._lr_scheduler.step()

            if self._gen_net_ema is not None:
                self._update_ema(self._ema_decay)

            return loss.detach().item(), pred_gt.detach()

    @torch.no_grad()
    def val_step(self, *args):
        lq, gt = args[0], args[1]
        pred_gt = self._gen_net_ema(lq) if self._gen_net_ema else self._gen_net(lq)
        loss = self._pix_criterion(pred_gt, gt)

        return loss.item(), pred_gt

    def get_model_states_dict(self):
        unwrapped_gen_net = (
            self._accelerator.unwrap_model(self._gen_net_ema)
            if self._gen_net_ema
            else self._accelerator.unwrap_model(self._gen_net)
        )
        return {"gen": unwrapped_gen_net.state_dict()}

    def get_last_learning_rate(self):
        return self._lr_scheduler.get_last_lr()[0]

    def train(self):
        self._gen_net.train()

    def eval(self):
        self._gen_net.eval()

    def _update_ema(self, decay):
        gen_net_params = dict(self._gen_net.named_parameters())
        gen_net_ema_params = dict(self._gen_net_ema.named_parameters())

        for name in gen_net_params.keys():
            gen_net_ema_params[name].data.mul_(decay).add_(
                gen_net_params[name].data, alpha=1 - decay
            )

    def _initialize_gen_net_ema(self):
        self._gen_net_ema = None
        if self._ema_decay > 0:
            self._gen_net_ema = copy.deepcopy(self._gen_net)
            for param in self._gen_net_ema.parameters():
                param.requires_grad_(False)
            self._gen_net_ema = self._accelerator.prepare(self._gen_net_ema)
            self._gen_net_ema.eval()


class SRGANWrapper(SRNetWrapper):
    def __init__(
        self,
        gen_net,
        disc_net,
        gen_opt,
        disc_opt,
        lr_scheduler,
        accelerator,
        loss_config,
        ema_decay=0,
    ):
        self._gen_net = gen_net
        self._disc_net = disc_net
        self._gen_opt = gen_opt
        self._disc_opt = disc_opt
        self._lr_scheduler = lr_scheduler
        self._accelerator = accelerator
        self._loss_config = loss_config
        self._ema_decay = ema_decay

        self._pix_criterion = (
            L1Loss(loss_config.PIXEL_LOSS.WEIGHT).to(accelerator.device)
            if loss_config.PIXEL_LOSS.TYPE == "l1"
            else MSELoss(loss_config.PIXEL_LOSS.WEIGHT).to(accelerator.device)
        )
        self._perc_criterion = PerceptualLoss(
            loss_config.PERCEPTUAL_LOSS.LAYER_WEIGHTS,
            loss_config.PERCEPTUAL_LOSS.VGG_TYPE,
            loss_config.PERCEPTUAL_LOSS.USE_INPUT_NORM,
            loss_config.PERCEPTUAL_LOSS.RANGE_NORM,
            loss_config.PERCEPTUAL_LOSS.WEIGHT,
            loss_config.PERCEPTUAL_LOSS.STYLE_WEIGHT,
            loss_config.PERCEPTUAL_LOSS.CRITERION,
        ).to(accelerator.device)
        self._gan_criterion = GANLoss(
            loss_config.GAN_LOSS.GAN_TYPE,
            loss_config.GAN_LOSS.REAL_LABEL_VAL,
            loss_config.GAN_LOSS.FAKE_LABEL_VAL,
            loss_config.GAN_LOSS.WEIGHT,
        ).to(accelerator.device)

        (
            self._gen_net,
            self._disc_net,
            self._gen_opt,
            self._disc_opt,
            self._lr_scheduler,
        ) = self._accelerator.prepare(
            self._gen_net,
            self._disc_net,
            self._gen_opt,
            self._disc_opt,
            self._lr_scheduler,
        )

        self._initialize_gen_net_ema()

    def train_step(self, *args, max_grad_norm=1):
        with self._accelerator.accumulate(self._gen_net):
            lq, gt, gt_usm = args

            pix_gt, perc_gt, gan_gt = gt, gt, gt
            if self._loss_config.PIXEL_LOSS.GT_USM:
                pix_gt = gt_usm
            if self._loss_config.PERCEPTUAL_LOSS.GT_USM:
                perc_gt = gt_usm
            if self._loss_config.GAN_LOSS.GT_USM:
                gan_gt = gt_usm

            gen_loss, pred_gt = self._optimize_gen(lq, pix_gt, perc_gt, max_grad_norm)
            disc_loss = self._optimize_disc(gan_gt, pred_gt, max_grad_norm)
            self._lr_scheduler.step()

            if self._gen_net_ema is not None:
                self._update_ema(self._ema_decay)

            return {"gen_loss": gen_loss, "disc_loss": disc_loss}, pred_gt

    @torch.no_grad()
    def val_step(self, *args):
        lq, gt = args[0], args[1]
        gen_loss, pred_gt = self._get_gen_loss(lq, gt, gt)
        disc_loss = self._get_disc_loss(gt, pred_gt)

        return {"gen_loss": gen_loss.item(), "disc_loss": disc_loss.item()}, pred_gt

    def get_model_states_dict(self):
        unwrapped_gen_net = (
            self._accelerator.unwrap_model(self._gen_net_ema)
            if self._gen_net_ema
            else self._accelerator.unwrap_model(self._gen_net)
        )
        unwrapped_disc_net = self._accelerator.unwrap_model(self._disc_net)
        return {
            "gen": unwrapped_gen_net.state_dict(),
            "disc": unwrapped_disc_net.state_dict(),
        }

    def train(self):
        self._gen_net.train()
        self._disc_net.train()

    def eval(self):
        self._gen_net.eval()
        self._disc_net.eval()

    def _optimize_gen(self, lq, pix_gt, perc_gt, max_grad_norm):
        self._gen_opt.zero_grad()
        gen_loss, pred_gt = self._get_gen_loss(lq, pix_gt, perc_gt)
        self._accelerator.backward(gen_loss)
        if self._accelerator.sync_gradients:
            self._accelerator.clip_grad_norm_(self._gen_net.parameters(), max_grad_norm)
        self._gen_opt.step()

        return gen_loss.detach().item(), pred_gt

    def _optimize_disc(self, gan_gt, pred_gt, max_grad_norm):
        self._disc_opt.zero_grad()
        disc_loss = self._get_disc_loss(gan_gt, pred_gt)
        self._accelerator.backward(disc_loss)
        if self._accelerator.sync_gradients:
            self._accelerator.clip_grad_norm_(
                self._disc_net.parameters(), max_grad_norm
            )
        self._disc_opt.step()

        return disc_loss.detach().item()

    def _get_gen_loss(self, lq, pix_gt, perc_gt):
        pred_gt = self._gen_net(lq)
        pixel_loss = self._pix_criterion(pred_gt, pix_gt)
        percep_loss, _ = self._perc_criterion(pred_gt, perc_gt)
        with torch.no_grad():
            fake_pred = self._disc_net(pred_gt)
        gen_gan_loss = self._gan_criterion(fake_pred, True, is_disc=False)
        total_gen_loss = pixel_loss + percep_loss + gen_gan_loss

        return total_gen_loss, pred_gt.detach()

    def _get_disc_loss(self, gan_gt, pred_gt):
        real_pred = self._disc_net(gan_gt)
        real_loss = self._gan_criterion(real_pred, True, is_disc=True)
        fake_pred = self._disc_net(pred_gt)
        fake_loss = self._gan_criterion(fake_pred, False, is_disc=True)
        total_disc_loss = real_loss + fake_loss

        return total_disc_loss
