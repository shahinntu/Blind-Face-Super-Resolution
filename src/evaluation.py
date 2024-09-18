import numpy as np
from tqdm import tqdm

from src import Predictor


class Evaluator:
    def __init__(self, model, device):
        self._predictor = Predictor(model, device)

    def evaluate(self, lq_images, hq_images):
        total_psnr = 0
        for lq_image, hq_image in tqdm(zip(lq_images, hq_images), desc="Evaluating"):
            hq_pred = self._predictor.predict(lq_image)
            total_psnr += self._calculate_psnr(hq_pred, hq_image)

        return {"psnr": total_psnr / len(lq_images)}

    def _calculate_psnr(self, img1, img2):
        img1_array = np.array(img1) / 255.0 * 255
        img2_array = np.array(img2) / 255.0 * 255
        mse_value = np.mean((img1_array - img2_array) ** 2)

        return 20.0 * np.log10(255.0 / np.sqrt(mse_value))
