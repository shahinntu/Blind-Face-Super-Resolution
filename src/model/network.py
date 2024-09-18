import os
import copy

import torch
import torch.nn as nn
from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.swinir_arch import SwinIR
from basicsr.archs.discriminator_arch import UNetDiscriminatorSN

from src import Params, LatticeNet, MobileSR, load_checkpoint


class AutoNetwork:
    @staticmethod
    def from_config(config):
        network_name = config.TYPE

        if network_name == "MSRResNet":
            network = MSRResNet(
                config.NUM_IN_CH,
                config.NUM_OUT_CH,
                config.NUM_FEAT,
                config.NUM_BLOCK,
                config.UPSCALE,
            )
        elif network_name == "RRDBNet":
            network = RRDBNet(
                config.NUM_IN_CH,
                config.NUM_OUT_CH,
                config.UPSCALE,
                config.NUM_FEAT,
                config.NUM_BLOCK,
                config.NUM_GROW_CH,
            )
        elif network_name == "UNetDiscriminatorSN":
            network = UNetDiscriminatorSN(
                config.NUM_IN_CH, config.NUM_FEAT, config.SKIP_CONNECTION
            )
        elif network_name == "SwinIR":
            network = SwinIR(
                upscale=config.UPSCALE,
                img_size=config.IMG_SIZE,
                window_size=config.WINDOW_SIZE,
                img_range=config.IMG_RANGE,
                depths=config.DEPTHS,
                embed_dim=config.EMBED_DIM,
                num_heads=config.NUM_HEADS,
                mlp_ratio=config.MLP_RATIO,
                upsampler=config.UPSAMPLER,
                resi_connection=config.RESI_CONNECTION,
            )
        elif network_name == "LatticeNet":
            network = LatticeNet(config.N_FEATS, config.UPSCALE)
        elif network_name == "MobileSR":
            network = MobileSR(
                config.N_FEATS, config.N_HEADS, config.RATIOS, config.UPSCALE
            )
        else:
            raise NameError(f"Network '{network_name}' not found.")

        return network

    @staticmethod
    def from_pretrained(log_dir, best_or_last="best"):
        config_path = os.path.join(log_dir, "configs", "config.json")
        state_path = os.path.join(log_dir, "state", f"gen_{best_or_last}.pth")

        config = Params.from_json_path(config_path)
        network = AutoNetwork.from_config(config.MODEL.GEN)
        load_checkpoint(state_path, network)

        return network


class InterpolatedNetwork(nn.Module):
    def __init__(self, net_a, net_b, lambda_val):
        super(InterpolatedNetwork, self).__init__()
        self.net_a = net_a
        self.net_b = net_b
        self.lambda_val = lambda_val
        self.interpolated_net = self.interpolate_networks()

    def interpolate_networks(self):
        interpolated_net = copy.deepcopy(self.net_a)

        with torch.no_grad():
            for param_a, param_b, param_interpolated in zip(
                self.net_a.parameters(),
                self.net_b.parameters(),
                interpolated_net.parameters(),
            ):
                param_interpolated.data.copy_(
                    self.lambda_val * param_a.data
                    + (1 - self.lambda_val) * param_b.data
                )

        return interpolated_net

    def forward(self, x):
        return self.interpolated_net(x)
