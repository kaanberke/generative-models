import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from typing import Optional, Tuple


class CustomDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 4,
        train_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the DataModule.
        
        Parameters:
            data_dir (str): Path to the data directory.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of worker threads for data loading.
            train_transforms (transforms.Compose, optional): Transformations for the training dataset.
            val_transforms (transforms.Compose, optional): Transformations for the validation dataset.
            test_transforms (transforms.Compose, optional): Transformations for the test dataset.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def prepare_data(self):
        """
        Download data, perform one-time processing, and save results.
        This method is only called on one GPU/node when using distributed training.
        """
        # TODO: Add data download/preparation logic here, if needed.
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Load data, split datasets, apply transformations, etc.
        This method is called on every GPU/node and should be used for "stateful" operations, such as splitting a dataset.
        
        Parameters:
            stage (str, optional): Either 'fit' (for train/val) or 'test', or None (for both).
        """
        # TODO: Add data setup logic here, e.g., load and split datasets, apply transformations, etc.
        pass

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        # TODO: Implement logic to return the train DataLoader.
        pass

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        # TODO: Implement logic to return the validation DataLoader.
        pass

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        # TODO: Implement logic to return the test DataLoader.
        pass
