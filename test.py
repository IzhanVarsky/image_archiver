import torch.cuda

from model_wrapper import ModelWrapper
from my_ae import MyAEArchiver


def run_testing(B=None, comp='high'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Params:")
    print("B:", B)
    model = MyAEArchiver(B=B, compression=comp)
    model_name = f"ImageAutoEncoder_B{B}_TestArch5"
    model = ModelWrapper(model, None, None, device, model_name)
    ckpt = 'train_logs/1693669446694805377/ImageAutoEncoder_BNone_TestArch5_all_imgs_sz256_best_metric_model.pth'
    model.load_checkpoint(ckpt)
    model.val_model('./test_images', './out_test_images')


if __name__ == '__main__':
    run_testing()
