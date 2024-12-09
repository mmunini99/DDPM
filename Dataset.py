import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, Subset



# Transformations applied on each image => bring them into a numpy array and normalize between -1 and 1 (as in the original DDPM paper
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = img / 255. - 1.0
    img = np.expand_dims(img, axis = -1)
    return img


# We need to stack the batch elements as numpy arrays
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

# override the PyTorch method to get only image
class MNISTWithoutLabels(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist_dataset = MNIST(root=root, train=train, transform=transform, download=download)
        
    def __getitem__(self, index):
        image, _ = self.mnist_dataset[index]  # We ignore the label (second value)
        return image
    
    def __len__(self):
        return len(self.mnist_dataset)


def load_data(DATASET_PATH, batch_size, width, heigth, bool_download, subset_num):
    train_transform = transforms.Compose([transforms.Resize((width, heigth)),
                                      image_to_numpy
                                     ])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNISTWithoutLabels(root=DATASET_PATH,
                                train=True,
                                transform=train_transform,
                                download=bool_download)
    # train_set, val_set = torch.utils.data.random_split(train_dataset,
    #                                                 [50000, 10000],
    #                                                 generator=torch.Generator().manual_seed(42))
    if subset_num is not None:
        train_dataset = Subset(train_dataset, list(range(subset_num)))

    # Loading the test set
    test_set = MNISTWithoutLabels(root=DATASET_PATH,
                            train=False,
                            transform=train_transform,
                            download=bool_download)

    # We define a set of data loaders that we can use for various purposes later.
    # Note that for actually training a model, we will use different data loaders
    # with a lower batch size.
    train_loader = data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate)
    # val_loader   = data.DataLoader(val_set,
    #                             batch_size=128,
    #                             shuffle=False,
    #                             drop_last=False,
    #                             collate_fn=numpy_collate)
    test_loader  = data.DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate)

    return train_loader, test_loader, 