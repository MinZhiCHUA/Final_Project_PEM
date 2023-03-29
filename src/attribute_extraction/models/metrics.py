from typing import List

import torch


def compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> List[torch.FloatTensor]:

    with torch.no_grad():
        predicted_label = output.argmax(dim=1)
        acc = (predicted_label == target).float().mean()

        return acc
