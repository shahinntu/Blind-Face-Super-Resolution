from torch.utils.data import Dataset

from PIL import Image


class FFHQDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, mode="train"):
        super().__init__()
        self._hf_dataset = hf_dataset
        self._transform = transform
        self._mode = mode

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, index):
        data = self._hf_dataset[index]

        gt_image = Image.open(data["gt_image_path"])
        if self._transform:
            gt_image = self._transform(gt_image)

        if self._mode == "train":
            return gt_image

        elif self._mode == "val":
            lq_image = Image.open(data["lq_image_path"])
            if self._transform:
                lq_image = self._transform(lq_image)

            return lq_image, gt_image
