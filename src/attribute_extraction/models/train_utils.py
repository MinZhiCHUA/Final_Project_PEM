import pathlib
import random
from typing import Iterator, List

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, Sampler

from attribute_extraction.utils.matching import preprocess_function


class AttributeDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        context_col_name: str,
        label_col_name: str,
        attribute_code_col_name: str,
        aug_p: float = 0.0,
    ):
        self.data = data
        self.context_col_name = context_col_name
        self.label_col_name = label_col_name
        self.attribute_code_col_name = attribute_code_col_name
        self.aug_p = aug_p

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, item):
        context = self.data.loc[item, self.context_col_name]
        label = self.data.loc[item, self.label_col_name]
        attribute_code = self.data.loc[item, self.attribute_code_col_name]

        return context, label, attribute_code


def collate_fun_generator(batch: List, tokenizer, max_len: int = 512):

    contexts, labels, attribute_codes = zip(*batch)

    assert len(set(attribute_codes)) == 1

    network_input = tokenizer(
        list(contexts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    network_input[attribute_codes[0]] = torch.tensor(labels)

    return network_input


def build_callbacks(
    output_path: pathlib.Path,
    model_name: str,
    early_patience: int = 20,
):

    monitor = "validation_loss"
    mode = "min"

    early_stopping = pytorch_lightning.callbacks.EarlyStopping(
        monitor=monitor, mode=mode, patience=early_patience
    )
    checkpoint = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=str(output_path),
        filename=model_name,
        save_weights_only=True,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(output_path), name=model_name, default_hp_metric=False
    )

    learning_rate_monitor = pytorch_lightning.callbacks.LearningRateMonitor()

    return [early_stopping, checkpoint, learning_rate_monitor], tensorboard_logger


class MultiAttributeBatchSampler(Sampler[List[int]]):
    def __init__(
        self, data: pd.DataFrame, batch_size: int, split_col: str, weight_col: str
    ) -> None:

        self.batch_size = batch_size
        self.col_val_to_indexes = {
            k: v.tolist() for k, v in data.groupby(split_col).indices.items()
        }
        self.n_elements = data.shape[0]
        self.weight_col = weight_col
        self.data = data

    def generate_batches(self):

        batches = []

        for col_val, indexes in self.col_val_to_indexes.items():

            weight_att = self.data[self.weight_col].iloc[indexes]
            batches.extend(
                list(weighted_random_chunks(indexes, weights=weight_att, n=self.batch_size))
            )

        random.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[List[int]]:

        batches = self.generate_batches()

        for batch_indexes in batches:
            yield batch_indexes

    def __len__(self) -> int:
        return int(
            sum(
                np.ceil(len(value) / self.batch_size) for value in self.col_val_to_indexes.values()
            )
        )

    def get_weights(self):
        self.data


def weighted_random_chunks(lst, weights, n):
    """Yield random n-sized chunks from lst."""

    re_sampled = random.choices(lst, weights=weights, k=len(lst))
    random.shuffle(re_sampled)

    for i in range(0, len(re_sampled), n):
        yield re_sampled[i : i + n]


class QADataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, item):
        question = self.data.loc[item, "question"]
        context = self.data.loc[item, "context"]
        start_end = self.data.loc[item, "start_end"]

        return preprocess_function(question, context, start_end, self.tokenizer)
