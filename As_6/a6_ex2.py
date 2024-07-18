import torch
from torch.utils.data import DataLoader
from a3_ex1 import ImagesDataset
from a6_ex1 import TransformedImagesDataset

def stacking(batch_as_list: list):
    trans_imgs = []
    trans_names = []
    indices = []
    class_ids = []
    class_names = []
    img_paths = []

    for item in batch_as_list:
        trans_img, trans_name, index, class_id, class_name, img_path = item
        trans_imgs.append(trans_img)
        trans_names.append(trans_name)
        indices.append(index)
        class_ids.append(class_id)
        class_names.append(class_name)
        img_paths.append(img_path)

    stacked_images = torch.stack(trans_imgs).to(torch.int32)
    # to match the .pdf output otherwise returns torch.float32
    stacked_indices = torch.Tensor(indices).unsqueeze(1).to(torch.int32)
    stacked_class_ids = torch.Tensor(class_ids).unsqueeze(1).to(torch.int32)

    return stacked_images, trans_names, stacked_indices, stacked_class_ids, class_names, img_paths

if __name__ == "__main__":
    dataset = ImagesDataset("./validated_images", 100, 100, int)
    transformed_ds = TransformedImagesDataset(dataset)
    dl = DataLoader(transformed_ds, batch_size=7, shuffle=False, collate_fn=stacking)
    for i, (images, trans_names, indices, classids, classnames, img_paths) in enumerate(dl):
        print(f'mini batch: {i}')
        print(f'images shape: {images.shape}, dtype={images.dtype}')
        print(f'trans_names: {trans_names}')
        print(f'indices: {indices}')
        print(f'class ids: {classids}')
        print(f'class names: {classnames}\n')