import torch
from torch.utils.data import DataLoader
from a3_ex1 import ImagesDataset

def stacking(batch_as_list: list):

    images, class_ids, class_names, image_filepaths = zip(*batch_as_list)
    stacked_images = torch.stack([torch.from_numpy(img) for img in images])
    stacked_class_ids = torch.tensor(class_ids, dtype=torch.int32).unsqueeze(1)
    class_names = list(class_names)
    image_filepaths = list(image_filepaths)

    return stacked_images, stacked_class_ids, class_names, image_filepaths

# if __name__=="__main__":
#     ds = ImagesDataset("./validated_images", 100, 100, int)
#     dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=stacking)
#     for i, (images, classids, classnames, image_filepaths) in enumerate(dl):
#         print(f'mini batch: {i}')
#         print(f'images shape: {images.shape}')
#         print(f'class ids: {classids}')
#         print(f'class names: {classnames}\n')