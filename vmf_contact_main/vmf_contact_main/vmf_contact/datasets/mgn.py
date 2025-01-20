import logging
from typing import Any, Optional

import torch
import torchvision.transforms.v2 as v2  # type: ignore
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from ._utils._base import DataModule, TransformedDataset
from ._utils._registry import register
from ._utils import dataset_train_test_split
from .dataset import MGNDataset, custom_collate_fn
import glob
logger = logging.getLogger(__name__)


@register("mgn")
class MGNDataModule(DataModule):
    """
    Data module for the NYU Depth v2 dataset.
    """

    def __init__(self, args, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(args.data_root_dir, seed)
        self.did_setup = False
        self.did_setup_ood = False
        self.args = args
        self.num_workers = args.num_workers
        self.image_size = args.image_size
        self.grid_num = (
            int(args.image_size[0] * args.scale // 14),
            int(args.image_size[1] * args.scale // 14),
        )

    @property
    def output_type(self):
        return "normal"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 480, 640])

    # @property
    # def gradient_accumulation_steps(self) -> int:
    #   return 2

    def prepare_data(self) -> None:
        # Download NYU Depth v2
        logger.info("Preparing MGN...")
        if isinstance(self.root, list):
            root_dir = []
            for r in self.root:
                root_dir += glob.glob(str(r))
        else:
            root_dir = glob.glob(str(self.root))

    def setup(
        self, stage: Optional[str] = None, crop_scale=0.6, theta_deg=15, no_rot=False
    ) -> None:
        # Random color augmentation, only applied to the image
        random_color_aug = v2.Compose(
            [
                # v2.RandomPhotometricDistort(p=0.3),
                v2.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.6, hue=0.3),
            ]
        )

        if not self.did_setup:
            train_data = MGNDataset(
                self.args.data_root_dir,
                # self.args.data_root_dir_debug,
                image_size=self.args.image_size,
                pcd_with_rgb=self.args.pcd_with_rgb,
                num_cameras=self.args.camera_num,
            )
            train, val = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )
            self.train_dataset = TransformedDataset(
                train,
                color_aug=random_color_aug,
                # joint_transform=SpatialTransform(input_h=input_h, input_w=input_w, crop_scale=crop_scale),
            )
            self.val_dataset = val
            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = MGNDataset(
                self.args.data_root_dir_test,
                image_size=self.args.image_size,
                pcd_with_rgb=self.args.pcd_with_rgb,
            )
            self.did_setup_ood = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            collate_fn=custom_collate_fn,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            collate_fn=custom_collate_fn,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            collate_fn=custom_collate_fn,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class SpatialTransform:
    def __init__(self, input_h=480, input_w=640, p: float = 0.5, crop_scale=0.6):
        self.p = p
        # Random crop and flip
        r = input_w / input_h
        self.random_spatial = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(input_h, input_w),
                    ratio=(r, r),
                    scale=(crop_scale, crop_scale),
                ),
            ]
        )

    def __call__(self, items):
        (items["rgb"], items["pcd_img"], items["depth"], items["instance"]) = (
            self.random_spatial(
                {
                    "rgb": items["rgb"],
                    "pcd_img": items["pcd_img"],
                    "depth": items["depth"],
                    "instance": items["instance"],
                }
            )
        )
        return items


def _noop(x: Any) -> Any:
    return x
