from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import ImageDataset


def get_dataloader_dataset(train_dir="",
                           test_dir="",
                           val_dir="",
                           batch_size=64,
                           num_workers=4,
                           sz=224):
    need_train = train_dir != ""
    need_test = test_dir != ""
    need_val = val_dir != ""
    image_datasets = dict()
    dataloaders = dict()
    train_data_transforms = T.Compose([
        T.Resize((sz, sz)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
    ])
    test_data_transforms = T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
    ])
    if need_train:
        image_datasets['train'] = ImageDataset(train_dir, transform=train_data_transforms)
        dataloaders['train'] = DataLoader(image_datasets['train'],
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          pin_memory=True)
    if need_test:
        image_datasets['test'] = ImageDataset(test_dir, transform=test_data_transforms)
        dataloaders['test'] = DataLoader(image_datasets['test'],
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=True)
    if need_val:
        image_datasets['val'] = ImageDataset(val_dir, transform=test_data_transforms)
        dataloaders['val'] = DataLoader(image_datasets['val'],
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)
    return dataloaders, image_datasets
