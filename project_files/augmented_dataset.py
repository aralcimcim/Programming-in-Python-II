import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def augment_image(img_np: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.RandomRotation((0, 45)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(100, 100), scale=(0.08, 1.0), antialias=True)
    ])
    
    trans_img = transform(img_np)
    trans_name = "randomly augment"
    
    return trans_img, trans_name

class AugmentedImagesDataset(Dataset):
    def __init__(self, data_set: Dataset):
        self.data_set = data_set

    def __getitem__(self, index: int):
        img_np, classid, classname, img_path = self.data_set[index % len(self.data_set)]
        trans_img, trans_name = augment_image(img_np)
        return trans_img, classid, classname, img_path, index, trans_name

    def __len__(self):
        return len(self.data_set)

#test
# if __name__ == "__main__":
#     test_data = augment_image(torch.rand(3, 100, 100))