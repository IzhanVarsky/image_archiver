import torch.cuda
from torch import nn, optim

from early_stopping import EarlyStopping
from loader_utils import get_dataloader_dataset
from model_wrapper import ModelWrapper
from my_ae import MyAEArchiver


def run_training(B=8, compression='high'):
    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_datasets_root = 'datasets'
    dataset_name = 'all_imgs'
    # dataset_name = 'mnist_only'
    train_dir = [
        f"./{all_datasets_root}/huge_imagenet512",
        f"./{all_datasets_root}/colorized-MNIST/train",
        f"./{all_datasets_root}/imagenet-mini/train",
    ]
    test_dir = [
        f"./{all_datasets_root}/imagenet-mini/val",
        f"./{all_datasets_root}/colorized-MNIST/test",
    ]

    img_size = 256
    dataloaders, image_datasets = get_dataloader_dataset(train_dir=train_dir,
                                                         test_dir=test_dir,
                                                         val_dir="",
                                                         batch_size=batch_size,
                                                         num_workers=4,
                                                         sz=img_size,
                                                         )
    print("Params:")
    print("batch_size:", batch_size)
    print("img_size:", img_size)
    print("train_dirs:", train_dir)
    print("test_dirs:", test_dir)
    print("train dataset length:", len(image_datasets['train']))
    print("B:", B)
    print("Compression:", compression)
    model = MyAEArchiver(B=B, compression=compression)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.4, verbose=True)
    model_name = f"ImageAutoEncoder_B{B}_Comp{compression.capitalize()}_changed_last_dec_block"
    model = ModelWrapper(model, optimizer, criterion, device, model_name, schedulers=[scheduler])

    checkpoint_name = f"{model.model_name}_{dataset_name}_sz{img_size}"
    early_stopping = EarlyStopping(model_name=checkpoint_name, save_best=True,
                                   use_early_stop=False, metric_decreasing=True)
    # ckpt = 'train_logs/1693694047975636286/ImageAutoEncoder_BNone_TestArch5_all_imgs_sz256_best_metric_model.pth'
    # model.load_checkpoint(ckpt)
    # print(f"Loaded ckpt: {ckpt}")
    model.train_model(dataloaders, early_stopping, num_epochs=20)


if __name__ == '__main__':
    run_training(B=None, compression='high')
    run_training(B=None, compression='low')
    # run_training(B=8)
    # run_training(B=2)
    # run_training(B=4)
    # run_training(B=1)
