import torch
import torchvision.transforms.functional as TF

from src import AutoNetwork, tensor2PIL


class Predictor:
    def __init__(self, model, device):
        self._model = model
        self._device = device

        self._model = self._model.to(self._device)
        self._model.eval()

    @classmethod
    def from_pretrained(cls, log_dir, best_or_last="best", device="cpu"):
        model = AutoNetwork.from_pretrained(log_dir, best_or_last)

        return cls(model, device)

    @torch.no_grad()
    def predict(self, image):
        img_tensor = TF.to_tensor(image).to(self._device)
        pred_tensor = self._model(img_tensor.unsqueeze(0))
        pred_img = tensor2PIL(pred_tensor.squeeze(0))

        return pred_img
