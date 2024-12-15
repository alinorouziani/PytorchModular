"""
This makes the data ready for the model, which gives you:
1.train_dataloader
2.test_dataloader
3.class_names
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size:int):
  """
  Creates training and testing Dataloader.
  Args:
  train_dir: The path to training directory.
  test_dir: The path to test directory.
  transform: Torchvision transforms to perform on training and testing data.
  batch_size: The number of samples per batch in each of the dataloaders.
  Returns:
  A tuple of (train_dataloader, test_dataloader, class_names).
  """
  train_data = datasets.ImageFolder(train_dir, transform = transform)
  test_data = datasets.ImageFolder(test_dir, transform = transform)
  
  class_names = train_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size = batch_size,
      shuffle = True,
      pin_memory = True
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size = batch_size,
      shuffle = False,
      pin_memory = True
  )

  return train_dataloader, test_dataloader, class_names
