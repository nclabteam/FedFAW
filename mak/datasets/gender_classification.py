import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms

class CelebAGenderDataset(Dataset):
    def __init__(self, root, split="train", transform=None, download=False):
        """
        Custom dataset for CelebA gender classification (0 = Female, 1 = Male).
        Args:
            root (str): Path to the dataset.
            split (str): 'train' or 'test'.
            transform (callable, optional): Transformations to apply to the images.
            download (bool): Whether to download the dataset.
        """
        self.dataset = datasets.CelebA(root=root, split=split, target_type="attr", download=download)
        self.transform = transform

        # Extract gender labels (21st attribute: Male = 1, Female = -1)
        self.targets = self.dataset.attr[:, 20]  
        
        # Convert Male=1, Female=-1 to Male=1, Female=0
        self.targets = (self.targets == 1).long()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore original labels
        label = self.targets[idx]  # Binary label: 0 (Female), 1 (Male)
        label = int(label)
        if self.transform:
            image = self.transform(image)

        return image, label