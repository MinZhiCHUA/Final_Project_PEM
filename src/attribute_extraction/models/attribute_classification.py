import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from attribute_extraction.models.metrics import compute_accuracy


class AttributeClassifier(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        freeze_backbone=False,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-4,
        warmup_steps: int = 500,
        estimated_stepping_batches: int = 5000,
        num_cycles: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not any(
                    name.startswith(cl)
                    for cl in [
                        "classifier",
                        "pre_classifier",
                        "roberta.encoder.layer.11",
                        "distilbert.transformer.layer.5",
                    ]
                ):
                    param.requires_grad = False

    def forward(self, input_ids, labels=None, **kwargs):
        output = self.model(input_ids=input_ids, labels=labels, **kwargs)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):

        loss, logits = self(**batch)

        top1 = compute_accuracy(logits, batch["labels"])

        self.log("train_loss", loss)

        self.log("train_accuracy", top1)

        return loss

    def validation_step(self, batch, batch_idx):

        loss, logits = self(**batch)

        top1 = compute_accuracy(logits, batch["labels"])

        self.log("validation_loss", loss)

        self.log("validation_accuracy", top1)

        return loss

    def test_step(self, batch, batch_idx):

        loss, logits = self(**batch)

        top1 = compute_accuracy(logits, batch["labels"])

        self.log("testing_loss", loss)

        self.log("testing_accuracy", top1)

        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.estimated_stepping_batches,
            num_cycles=self.hparams.num_cycles,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class Pooler(torch.nn.Module):
    """
    Pooler
    """

    def __init__(self, input_dim, projection_dim=128, dropout=0.1):
        super().__init__()

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=projection_dim // 2),
            torch.nn.Dropout(dropout),
        )

        self.gru = torch.nn.GRU(
            input_size=projection_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=1,
            hidden_size=projection_dim // 2,
        )

        self.multi_head_att = torch.nn.MultiheadAttention(
            embed_dim=projection_dim, num_heads=8, dropout=dropout
        )

    def forward(self, batch):
        """
        batch : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        pooled: size (N, H)
        """

        batch_proj = self.projection(batch)

        batch_gru, _ = self.gru(batch_proj)

        batch_proj_permuted = batch_gru.permute(1, 0, 2)

        output, _ = self.multi_head_att(
            batch_proj_permuted, batch_proj_permuted, batch_proj_permuted
        )

        output = output.permute(1, 0, 2)

        return output[:, 0, :]


class ClassificationHead(torch.nn.Module):
    """
    ClassificationHead
    """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, output_dim),
        )

    def forward(self, batch):
        """
        batch : size (N, H), N: batch size, H: Hidden dimension
        pooled: size (N, num_classes)
        """
        return self.classification_head(batch)


class MultiAttributeClassifier(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        class_config: dict,
        freeze_backbone: bool = False,
        projection_dim: int = 128,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-4,
        warmup_steps: int = 500,
        estimated_stepping_batches: int = 5000,
        num_cycles: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.class_config = class_config
        self.projection_dim = projection_dim

        self.encoder = nn.Embedding(vocab_size, self.projection_dim)

        self.classification_heads = nn.ModuleDict(
            {
                output_name: nn.Sequential(
                    Pooler(
                        input_dim=self.projection_dim,
                        projection_dim=self.projection_dim,
                        dropout=dropout,
                    ),
                    ClassificationHead(
                        input_dim=self.projection_dim,
                        output_dim=len(output_params["id2label"]),
                        dropout=dropout,
                    ),
                )
                for output_name, output_params in self.class_config.items()
            }
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, output_names, attention_mask=None):

        src = self.encoder(input_ids)

        outputs = {name: self.classification_heads[name](src) for name in output_names}

        return outputs

    def predict(self, input_ids, output_names, attention_mask=None, top_k: int = 5):

        outputs = self.forward(
            input_ids=input_ids, output_names=output_names, attention_mask=attention_mask
        )

        predictions = {}

        for name in output_names:
            scores, indexes = torch.topk(
                functional.softmax(outputs[name], dim=1),
                k=min(top_k, len(self.class_config[name]["label2id"])),
            )

            predictions_class = []

            for sample_scores, sample_indexes in zip(
                scores.cpu().tolist(), indexes.cpu().tolist()
            ):

                predictions_class.append(
                    [
                        {"label": self.class_config[name]["id2label"][index], "score": score}
                        for score, index in zip(sample_scores, sample_indexes)
                    ]
                )

            predictions[name] = predictions_class

        return predictions

    def training_step(self, batch, batch_idx):

        target = [key for key in self.class_config if key in batch][0]

        outputs = self(
            batch["input_ids"], output_names=[target], attention_mask=batch["attention_mask"]
        )
        loss = self.ce_loss(outputs[target], batch[target])
        accuracy = compute_accuracy(outputs[target], batch[target])

        self.log("train_loss", loss)
        self.log(f"train_accuracy_{target}", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):

        target = [key for key in self.class_config if key in batch][0]

        outputs = self(
            batch["input_ids"], output_names=[target], attention_mask=batch["attention_mask"]
        )
        loss = self.ce_loss(outputs[target], batch[target])
        accuracy = compute_accuracy(outputs[target], batch[target])

        self.log("validation_loss", loss)
        self.log(f"validation_accuracy_{target}", accuracy)

        return loss

    def test_step(self, batch, batch_idx):

        target = [key for key in self.class_config if key in batch][0]

        outputs = self(
            batch["input_ids"], output_names=[target], attention_mask=batch["attention_mask"]
        )
        loss = self.ce_loss(outputs[target], batch[target])
        accuracy = compute_accuracy(outputs[target], batch[target])

        self.log("test_loss", loss)
        self.log(f"test_accuracy_{target}", accuracy)

        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters()],
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.estimated_stepping_batches,
            num_cycles=self.hparams.num_cycles,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
