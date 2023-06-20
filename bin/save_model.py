import json
from pathlib import Path

import bentoml
import torch
from loguru import logger
from transformers import AutoTokenizer

from attribute_extraction.models.attribute_classification import MultiAttributeClassifier
from attribute_extraction.models.mapper import Mapper

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="model_path", required=True)
    args = parser.parse_args()

    path = Path(args.model_path)

    hyper_parameters = json.load(open(path / "hyper_parameters.json"))

    tokenizer = AutoTokenizer.from_pretrained(hyper_parameters["model_name"])

    saved_tokenizer = bentoml.picklable_model.save_model(
        "attribute-classification-tokenizer",
        tokenizer,
        signatures={"__call__": {"batchable": False}},
    )

    logger.info(f"tokenizer saved: {saved_tokenizer}")

    mapper = Mapper.load(path / "mapper.json")

    model = MultiAttributeClassifier(
        vocab_size=tokenizer.vocab_size,
        class_config=mapper.mappings,
        projection_dim=hyper_parameters["projection_dim"],
    )

    model.load_state_dict(
        torch.load(
            path / "model.ckpt",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )["state_dict"]
    )

    model.eval()

    saved_model = bentoml.picklable_model.save_model(
        "attribute-classification-model",
        model,
        signatures={"predict_masked": {"batchable": False}},
    )

    logger.info(f"Model saved: {saved_model}")