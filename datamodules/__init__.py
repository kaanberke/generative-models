from pathlib import Path
from typing import Any, Callable, Optional

import lightning.pytorch as pl
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class RecursiveImageDataset(Dataset):

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        """
        Initialize the dataset. Store the file paths and set up transformations.
        
        Parameters:
            root_dir (str): Path to the root directory containing images.
            transform (Callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = Path(
            root_dir)  # Convert str path to pathlib.Path object
        self.transform = transform

        self.IMAGE_EXTENSIONS = [
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dng", ".webp"
        ]
        # Recursively get all image file paths
        self.image_paths = [
            image_path for image_path in self.root_dir.rglob("*.*")
            if image_path.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        """
        Get an item from the dataset.

        Parameters:
            idx (int): Index of the item.

        Returns:
            Any: The image and associated data (if available).
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class CustomDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 train_transforms: Optional[transforms.Compose] = None,
                 val_test_transforms: Optional[transforms.Compose] = None,
                 train_val_test_split: tuple = (0.7, 0.15, 0.15),
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_test_transforms = val_test_transforms
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # Create a dataset with all data
        full_dataset = RecursiveImageDataset(root_dir=self.data_dir,
                                             transform=self.train_transforms)

        # Calculate splits
        full_length = len(full_dataset)
        train_length = int(full_length * self.train_val_test_split[0])
        val_length = int(full_length * self.train_val_test_split[1])
        test_length = full_length - train_length - val_length

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_length, val_length, test_length])

        # Apply transformations for validation and test sets
        self.val_dataset.dataset.transform = self.val_test_transforms
        self.test_dataset.dataset.transform = self.val_test_transforms

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


# Testing the DataModule
if __name__ == "__main__":
    # Set the data path
    data_path = "data/processed/"

    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Instantiate the DataModule
    data_module = CustomDataModule(data_dir=data_path,
                                   batch_size=32,
                                   train_transforms=train_transforms,
                                   val_test_transforms=val_test_transforms,
                                   train_val_test_split=(0.7, 0.15, 0.15),
                                   num_workers=4)

    # Setup the DataModule (load and split the data)
    data_module.setup()

    # Visualize a batch of images from the training set
    train_loader = data_module.train_dataloader()

    # Get a batch of images
    images = next(iter(train_loader))

    # Create a grid of images and plot
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Sample Batch of Images from Training Set")
    plt.show()
