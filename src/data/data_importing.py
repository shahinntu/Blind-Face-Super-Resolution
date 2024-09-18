import os

import pandas as pd
from datasets import Dataset as HFDataset

from src import load_txt_to_list


def create_hf_train_dataset(train_data_dir):
    gt_dir = os.path.join(train_data_dir, "GT")
    meta_info_list = load_txt_to_list(
        os.path.join(train_data_dir, "meta_info_FFHQ5000sub_GT.txt")
    )
    meta_info_list = [meta_info.strip().split() for meta_info in meta_info_list]
    meta_info_dict = {
        filename: tuple(map(int, res_str.strip("()").split(",")))
        for filename, res_str in meta_info_list
    }

    data = []

    for filename in os.listdir(gt_dir):
        gt_image_path = os.path.join(gt_dir, filename)
        resolution = meta_info_dict.get(filename, None)

        if resolution is not None:
            data.append({"gt_image_path": gt_image_path, "resolution": resolution})

    data_df = pd.DataFrame(data)
    hf_dataset = HFDataset.from_pandas(data_df)

    return hf_dataset


def create_hf_val_dataset(val_data_dir):
    gt_dir = os.path.join(val_data_dir, "GT")
    lq_dir = os.path.join(val_data_dir, "LQ")

    data = []

    for filename in os.listdir(gt_dir):
        gt_image_path = os.path.join(gt_dir, filename)
        lq_image_path = os.path.join(lq_dir, filename)

        if os.path.exists(lq_image_path):
            data.append(
                {"gt_image_path": gt_image_path, "lq_image_path": lq_image_path}
            )

    data_df = pd.DataFrame(data)
    hf_dataset = HFDataset.from_pandas(data_df)

    return hf_dataset


def create_hf_test_dataset(test_data_dir):
    lq_dir = os.path.join(test_data_dir, "LQ")

    data = []

    for filename in os.listdir(lq_dir):
        lq_image_path = os.path.join(lq_dir, filename)

        if os.path.exists(lq_image_path):
            data.append({"lq_image_path": lq_image_path})

    data_df = pd.DataFrame(data)
    hf_dataset = HFDataset.from_pandas(data_df)

    return hf_dataset
