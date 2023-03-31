from functools import partial
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attribute_extraction.models.attribute_classification import MultiAttributeClassifier
from attribute_extraction.models.train_utils import AttributeDataset


def inference_collate_fun_generator(batch: List, tokenizer, max_len: int = 512):

    contexts, _, attribute_codes = zip(*batch)

    assert len(set(attribute_codes)) == 1

    network_input = tokenizer(
        list(contexts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    return network_input


def predict_attribute_dataset(
    model: MultiAttributeClassifier,
    tokenizer,
    dataframe: pd.DataFrame,
    context_col_name: str,
    attribute_code_col_name: str,
    attribute_code: str,
    batch_size=32,
    max_len=512,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()

    dataset = AttributeDataset(
        dataframe,
        context_col_name=context_col_name,
        label_col_name=context_col_name,
        attribute_code_col_name=attribute_code_col_name,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(inference_collate_fun_generator, tokenizer=tokenizer, max_len=max_len),
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader):
            outputs = model.predict(
                **{k: v.to(device) for k, v in batch.items()},
                output_names=[attribute_code],
            )

            predictions += outputs[attribute_code]

    return predictions
