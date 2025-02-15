import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import kornia.augmentation as K


class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def further_process_labels(labels):
    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()
    return labels


def corner_crop_extension(image):
    images = [F.crop(image, 0, 0, 256, 256), F.crop(image, 0, 256, 256, 256),
              F.crop(image, 256, 0, 256, 256), F.crop(image, 256, 256, 256, 256)]
    return images


def preprocess_train_data(data, final_label_processing=True):
    label = data[1]
    label = label.unsqueeze(0) if len(label.shape) == 2 else label
    # Treat image_and_label as 3-channel array
    image_and_label = torch.cat((data[0], label)).squeeze()

    image_and_label = K.RandomCrop((256, 256))(image_and_label).squeeze()
    image_and_label = K.RandomHorizontalFlip()(image_and_label).squeeze()
    image_and_label = K.RandomVerticalFlip()(image_and_label).squeeze()

    image = image_and_label[:2]
    image = K.Normalize([0.6851, 0.5235], [0.0820, 0.1102])(image).squeeze()

    label = image_and_label[2:].squeeze()

    if final_label_processing:
        further_process_labels(label)

    return image, label


def preprocess_train_data_uc(data):
    image, label = preprocess_train_data(data, final_label_processing=False)
    return image, label


def preprocess_eval_image(image):
    image = K.Normalize([0.6851, 0.5235], [0.0820, 0.1102])(image).squeeze()

    # Convert to PIL for easier transforms
    im_c1 = Image.fromarray(image[0].numpy()).resize((512, 512))
    im_c2 = Image.fromarray(image[1].numpy()).resize((512, 512))

    im_c1s = corner_crop_extension(im_c1)
    im_c2s = corner_crop_extension(im_c2)

    images = [torch.stack((transforms.ToTensor()(x).squeeze(),
                           transforms.ToTensor()(y).squeeze()))
              for (x, y) in zip(im_c1s, im_c2s)]
    images = torch.stack(images)

    return images


def preprocess_eval_label(label):
    label = label.numpy()
    label = Image.fromarray(label.squeeze()).resize((512, 512))
    labels = corner_crop_extension(label)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)
    further_process_labels(labels)

    return labels


def preprocess_eval_data(data):
    image, label = data[0].clone(), data[1].clone()

    images = preprocess_eval_image(image)
    labels = preprocess_eval_label(label)

    return images, labels
