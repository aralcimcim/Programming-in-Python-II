import numpy as np
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
from a3_ex1 import ImagesDataset


def augment_image(img_np: np.ndarray, index: int) -> torch.Tensor:

    trans_names = ["Gaussian Blur", 
                   "Random Rotation",  
                   "RandomVerticalFlip", 
                   "RandomHorizontalFlip", 
                   "ColorJitter"]
    
    transform_list = [transforms.GaussianBlur((5,9), (0.1,5)),
                      transforms.RandomRotation((0, 180)),
                      transforms.RandomVerticalFlip(1),
                      transforms.RandomHorizontalFlip(1),
                      transforms.ColorJitter(0.5, 0.3)]
    
    v = index % 7
    if v == 0:
        trans_img = torch.from_numpy(img_np)
        trans_name = "Original"

    elif v >= 1 and v <= 5:
        transform = transform_list[v - 1]
        trans_img = transform(torch.from_numpy(img_np))
        trans_name = trans_names[v - 1]

    elif v == 6:
        selected = random.sample(transform_list, 3)
        composed = transforms.Compose(selected)
        trans_img = composed(torch.from_numpy(img_np))
        trans_name = "Compose"

    return trans_img, trans_name

class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: Dataset):
        self.data_set = data_set

    def __getitem__(self, index: int):
        img_np, classid, classname, img_path = self.data_set[index // 7 % len(self.data_set)]
        trans_img, trans_name = augment_image(img_np, index)
        return trans_img, trans_name, index, classid, classname, img_path
    
    def __len__(self):
        return len(self.data_set) * 7

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dataset = ImagesDataset("./validated_images", 100, 100, int)
    transformed_ds = TransformedImagesDataset(dataset)
    fig, axes = plt.subplots(2, 4)

    for i in range(0, 8):
        trans_img, trans_name, index, classid, classname, img_path = transformed_ds.__getitem__(i)
        _i = i // 4
        _j = i % 4
        # int64 gives an error, converting to float
        axes[_i, _j].imshow(transforms.functional.to_pil_image(trans_img / 255), cmap='gray')
        axes[_i, _j].set_title(f'{trans_name}\n{classname}')

    fig.tight_layout()
    plt.show()
