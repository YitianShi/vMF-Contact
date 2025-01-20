# pylint: disable=abstract-method
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from . import IdentityScaler

DEFAULT_ROOT = Path.home() / "Research/simulation_isaaclab_rl"


class DataModule(pl.LightningDataModule, ABC):
    """
    Simple extension of PyTorch Lightning's data module which provides the input dimension and
    further details.
    """

    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    test_dataset: Dataset[Any]

    def __init__(self, root, seed: Optional[int] = None):
        super().__init__()
        if isinstance(root, str):
            self.root = Path(root or DEFAULT_ROOT)
        elif isinstance(root, list):
            self.root = [Path(r) for r in root]
        self.output_scaler = IdentityScaler()
        self.ood_datasets: Dict[str, Dataset[Any]] = {}
        self.generator = torch.Generator()
        if seed is not None:
            self.generator = self.generator.manual_seed(seed)

    @property
    @abstractmethod
    def output_type(self):
        """
        Returns the likelihood distribution for the outputs of this data module.
        """

    @property
    @abstractmethod
    def input_size(self) -> torch.Size:
        """
        Returns the size of the data items yielded by the data module.
        """

    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes if the data module yields training data with categorical
        outputs.
        """
        raise NotImplementedError

        # @property
        # def gradient_accumulation_steps(self) -> int:
        """
        Returns the number of batches from which to accumulate the gradients for training.
        """
        # return 1

    def transform_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Transforms a model's output such that the values are in the space of the true targets.
        This is, for example, useful when targets have been transformed for training.

        Args:
            output: The model output to transform.

        Returns:
            The transformed output.
        """
        return self.output_scaler.inverse_transform(output)

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        """
        Returns a set of dataloaders that can be used to measure out-of-distribution detection
        performance.

        Returns:
            A mapping from out-of-distribution dataset names to data loaders.
        """
        return {}


class TransformedDataset(Dataset[Any]):
    """
    Dataset that applies a transformation to its input and/or outputs.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        color_aug: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.dataset = dataset
        self.color_aug = color_aug or _noop
        self.target_transform = target_transform or _noop

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> Any:
        input_sample, gt_sample = self.dataset[index]
        input_sample["rgb"] = self.color_aug(input_sample["rgb"])
        input_sample = self.target_transform(input_sample)
        input_sample["rgb"] = torch.clamp(input_sample["rgb"], 0, 255)
        return input_sample, gt_sample


def _noop(x: Any) -> Any:
    return x
