import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


transform_train = A.Compose([
    A.Flip(),
    A.RandomRotate90(),
    A.Normalize(mean=[0.33875653, 0.36236152, 0.33635044], std=[0.13994731, 0.13818924, 0.14388752]),
    ToTensorV2()
])

transform_eval = A.Compose([
    A.Normalize(mean=[0.33896482, 0.362971, 0.33739454], std=[0.14057183, 0.13951756, 0.1442995]),
    ToTensorV2()
])


class PotsdamDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, i):
        image, label = self.data_list[i]
        image, label = np.array(image), np.array(label)
        transformed = self.transform(image=image, mask=label)
        image = transformed["image"]
        label = transformed["mask"]
        return image, label

    def __len__(self):
        return len(self.data_list)


def determine_mean_and_std(dataset):
    images = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        images.append(np.array(image))
    images = np.array(images)
    mean, std = np.mean(images, (0, 2, 3)), np.std(images, (0, 2, 3))
    return mean, std
