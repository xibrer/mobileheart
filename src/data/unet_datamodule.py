from typing import Any, Dict, Optional, Tuple
import os
import numpy as np
# from thop import profile
import scipy.io as io
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from lightning import LightningDataModule
from src.data.components.scg_datamodule import MyDataset


class UNETDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

        The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
        It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
        fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
        while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
        technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
        mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

        A `LightningDataModule` implements 7 key methods:

        ```python
            def prepare_data(self):
            # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
            # Download data, pre-process, split, save to disk, etc...

            def setup(self, stage):
            # Things to do on every process in DDP.
            # Load data, set variables, etc...

            def train_dataloader(self):
            # return train dataloader

            def val_dataloader(self):
            # return validation dataloader

            def test_dataloader(self):
            # return test dataloader

            def predict_dataloader(self):
            # return predict dataloader

            def teardown(self, stage):
            # Called on every process in DDP.
            # Clean up after fit or test.
        ```

        This allows you to share a full dataset without explaining how to download,
        split, transform and process the data.

        Read the docs:
            https://lightning.ai/docs/pytorch/latest/data/datamodule.html
        """

    def __init__(self,
                 data_dir: str = "data/",
                 train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
                 batch_size: int = 64,
                 user: str = "wxw",
                 num_workers: int = 0,
                 pin_memory: bool = False, ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.user = user
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def data_length(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 640

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # MyDataset(self.hparams.data_dir, train=True, download=True)
        # MyDataset(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MyDataset(self.hparams.data_dir, self.user, train=True)
            # testset = MyDataset(self.hparams.data_dir, self.user, train=False)
            dataset = ConcatDataset(datasets=[trainset])
            # print("***************************************",)
            split_length = [int(i * len(trainset)) for i in self.hparams.train_val_test_split]
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=split_length,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = UNETDataModule()