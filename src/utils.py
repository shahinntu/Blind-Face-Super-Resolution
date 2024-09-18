import os
import json
import logging
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import evaluate
from PIL import Image
import torch.nn.functional as F


class DotDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __contains__(self, item):
        return item in self.keys()


class ConfigBase:
    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
        for key, value in params.items():
            setattr(self, key, self._to_dotdict(value))


class Params(ConfigBase):
    def __init__(self, dict=None):
        if dict:
            self.__dict__ = self._to_dotdict(dict)

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            dict = json.load(f)
        return cls(dict)

    def _to_dotdict(self, obj):
        if isinstance(obj, dict):
            return DotDict({k: self._to_dotdict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._to_dotdict(item) for item in obj]
        else:
            return obj

    def add(self, key, value):
        if "ADDED" not in self.__dict__:
            self.ADDED = DotDict()
        self.ADDED[key] = value


class RunningAverageDict:
    def __init__(self):
        self.total_dict = {}
        self.steps = 0

    def update(self, val_dict):
        flattened_val_dict = self._flatten_dict(val_dict)
        for key, value in flattened_val_dict.items():
            if key not in self.total_dict:
                self.total_dict[key] = 0
            if key.endswith(":c"):
                self.total_dict[key] = value
            else:
                self.total_dict[key] += value
        self.steps += 1

    def serialize(self):
        keys = list(self.total_dict.keys())
        values = torch.tensor([list(self.total_dict.values())], dtype=torch.float32)
        steps = torch.tensor([self.steps], dtype=torch.float32)

        return keys, values, steps

    def reset(self):
        for key in self.total_dict:
            self.total_dict[key] = 0
        self.steps = 0

    def __call__(self):
        return {
            key.split(":")[0] if key.endswith(":c") else key: (
                value if key.endswith(":c") else value / float(self.steps)
            )
            for key, value in self.total_dict.items()
        }

    def _flatten_dict(self, d, sep=":"):
        items = []
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, sep=sep).items())
            else:
                items.append((k, v))
        return dict(items)


class RougeScore:
    def __init__(self, tokenizer, rouge_type):
        self._tokenizer = tokenizer
        self._rouge_type = rouge_type

        self._rouge = evaluate.load("rouge")

    def __call__(self, predictions, labels):
        decoded_predictions = self._tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels[labels == -100] = self._tokenizer.pad_token_id
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_score_dict = self._rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_aggregator=True,
            use_stemmer=True,
            rouge_types=[self._rouge_type],
        )
        return rouge_score_dict[self._rouge_type]


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def clear_handlers(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def save_checkpoint(state, is_best, checkpoint, network_type):
    filepath = os.path.join(checkpoint, f"{network_type}_last.pth")
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f"{network_type}_best.pth"))


def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist at {checkpoint}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        json.dump(d, f, indent=4)


def load_json_to_dict(json_path):
    with open(json_path, "r") as f:
        dict = json.load(f)

    return dict


def load_txt_to_list(txt_path):
    list = []
    with open(txt_path) as f:
        for line in f:
            list.append(line)

    return list


def save_list_to_txt(list, txt_path):
    with open(txt_path, "w") as f:
        for item in list:
            f.write(f"{item}\n")


def create_attribute_dict(file_path):
    attribute_dict = {}

    with open(file_path, "r") as file:
        next(file)
        next(file)

        for line in file:
            attribute_name, attribute_type = line.strip().split()
            attribute_type = int(attribute_type) - 1

            if attribute_type not in attribute_dict:
                attribute_dict[attribute_type] = [attribute_name]
            else:
                attribute_dict[attribute_type].append(attribute_name)

    return attribute_dict


def denormalize_batch(tensor_batch, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)

    tensor_batch = tensor_batch * std + mean
    return tensor_batch


def show_tensor_images(tensors, num=4):
    tensors = tensors.detach().cpu()
    tensors = tensors.permute(0, 2, 3, 1)
    tensors = tensors.clamp(0, 1)

    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(tensors[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def tensor2PIL(img_tensor):
    img_tensor = img_tensor.detach().cpu()
    img_tensor = img_tensor.permute(1, 2, 0)
    img_tensor = img_tensor.clamp(0, 1)
    img = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8))

    return img


def get_label_class_weights(labels):
    labels = torch.tensor(labels)

    class_weights = []
    for i in range(labels.size(1)):
        label = labels[:, i]
        class_counts = torch.bincount(label)
        class_frequencies = class_counts.float() / len(label)
        weight = 1.0 / class_frequencies
        weight = weight / weight.sum() * len(class_counts)

        class_weights.append(weight)

    return class_weights


def psnr(pred, gt):
    mse = F.mse_loss(pred, gt, reduction="mean")
    max_i = 1.0
    psnr = 20 * torch.log10(max_i / torch.sqrt(mse))
    return psnr.item()
