import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

import img_utils
from analyze_results import test_psnrs


def copy_py_files(folder_from, folder_to):
    os.makedirs(folder_to, exist_ok=True)
    for f in os.listdir(folder_from):
        if f.endswith(".py"):
            src = os.path.join(folder_from, f)  # путь к исходному файлу
            dst = os.path.join(folder_to, f)  # путь к целевому файлу
            shutil.copyfile(src, dst)  # копирование файла


class ModelWrapper(nn.Module):
    def __init__(self, model, optim, criterion, device, model_name, schedulers=[]):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = optim
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.schedulers = schedulers

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def logger_func(self, msg):
        print(msg)

    def run_epoch(self, phase, dataloader):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        all_elems_count = 0
        cur_tqdm = tqdm(dataloader)
        for inputs in cur_tqdm:
            bz = inputs.shape[0]
            all_elems_count += bz

            inputs = inputs.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * bz
            show_dict = {'Loss': f'{loss.item():.6f}'}
            cur_tqdm.set_postfix(show_dict)

        epoch_loss = running_loss / all_elems_count
        return epoch_loss

    @torch.no_grad()
    def test_epoch(self, dataloader):
        return self.run_epoch('test', dataloader)

    def train_epoch(self, dataloader):
        return self.run_epoch('train', dataloader)

    def train_model(self, dataloaders, early_stopping, num_epochs=5):
        log_folder = f'train_logs/{time.time_ns()}'
        out_log_file = 'train_log.txt'

        def logger_func(*msgs):
            print(*msgs)
            with open(f'{log_folder}/{out_log_file}', 'a+') as f:
                f.write(" ".join(msgs) + "\n")

        self.logger_func = logger_func

        os.makedirs(log_folder, exist_ok=True)
        copy_py_files('.', f'{log_folder}/py_files')

        out_batch_results = f'{log_folder}/out_batch_results'
        os.makedirs(out_batch_results, exist_ok=True)
        self.logger_func(f"Cur logging folder: {log_folder}")
        self.logger_func(f"Training model {self.model_name} with params:")
        self.logger_func(f"Optim: {self.optimizer}")
        self.logger_func(f"Criterion: {self.criterion}")

        saved_epoch_losses = {'train': [], 'test': []}
        saved_psnrs = dict()

        for epoch in range(1, num_epochs + 1):
            start_time = datetime.now()

            self.logger_func("=" * 100)
            self.logger_func(f'Epoch {epoch}/{num_epochs}')
            self.logger_func('-' * 10)

            for phase in ['train', 'test']:
                self.logger_func("--- Cur phase:", phase)
                epoch_loss = self.train_epoch(dataloaders[phase]) if phase == 'train' \
                    else self.test_epoch(dataloaders[phase])
                saved_epoch_losses[phase].append(epoch_loss)
                self.logger_func(f'{phase} loss: {epoch_loss:.6f}')
                if phase == 'train':
                    for scheduler in self.schedulers:
                        scheduler.step()
                if phase == 'test':
                    self.model.eval()
                    first_batch = next(iter(dataloaders[phase]))
                    with torch.no_grad():
                        first_batch = first_batch.to(self.device)
                        x_hat = self.model(first_batch)

                    limit = min(len(first_batch), 8)
                    fig, axes = plt.subplots(2, limit, figsize=(limit * 2, 4))
                    for i in range(limit):
                        axes[0, i].axis("off")
                        axes[1, i].axis("off")
                        axes[0, i].imshow(first_batch[i].cpu().permute(1, 2, 0).numpy())
                        axes[1, i].imshow(torch.clip(x_hat[i].cpu().permute(1, 2, 0), 0, 1).numpy())

                    plt.savefig(f"{out_batch_results}/epoch={epoch}.png")
                    plt.close('all')

            self.model.eval()
            plt.title('Losses during training')
            plt.plot(range(1, epoch + 1), saved_epoch_losses['train'], label='Train Loss')
            plt.plot(range(1, epoch + 1), saved_epoch_losses['test'], label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel(self.criterion.__class__.__name__)
            plt.legend(loc="upper left")
            plt.savefig(f'{log_folder}/loss_graph.png')
            plt.close('all')

            end_time = datetime.now()
            epoch_time = (end_time - start_time).total_seconds()
            self.logger_func("-" * 10)
            self.logger_func(f"Epoch Time: {math.floor(epoch_time // 60)} min {math.floor(epoch_time % 60)} sec")

            self.logger_func("Some evaluations...")
            self.val_model('./test_images', f'./{log_folder}/out_test_images_{epoch}')
            self.val_model('./datasets/imagenet-mini/train', f'./{log_folder}/out_imagenet-mini_images_train_{epoch}',
                           cnt=10)
            self.val_model('./datasets/imagenet-mini/val', f'./{log_folder}/out_imagenet-mini_images_val_{epoch}',
                           cnt=10)
            # self.val_model('./datasets/colorized-MNIST/train',
            #                f'./{log_folder}/out_colorized-MNIST_images_train_{epoch}',
            #                cnt=10)
            # self.val_model('./datasets/colorized-MNIST/test', f'./{log_folder}/out_colorized-MNIST_images_test_{epoch}',
            #                cnt=10)

            for root, dirs, files in os.walk('./test_images'):
                for file in files:
                    if img_utils.is_image_path(file):
                        img_path = os.path.join(root, file)
                        self.logger_func(f"Current Image: {img_path}")
                        predicted_pil_img = self.predict_image(img_path)
                        psnr = img_utils.get_psnr(predicted_pil_img, img_path)
                        saved_psnrs.setdefault(file, [])
                        saved_psnrs[file].append(psnr)
                        plt.plot(range(1, epoch + 1), saved_psnrs[file], label=file)
            plt.title('PSNR during training')
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.legend(loc="upper left")
            plt.savefig(f'{log_folder}/psnr_graph.png')
            plt.close('all')

            self.logger_func("Evaluations ended")

            early_stopping(saved_epoch_losses['test'][-1], self.model, out_dir=log_folder)

            self.logger_func("Evaluating PSNRs...")
            model_ckpt_name = f'{log_folder}/{early_stopping.get_checkpoint_name()}'
            test_dir = 'test_images'
            cur_out_log_folder = f'{log_folder}/testing_psnr_epoch_{epoch}'
            for B in [1, 2, 4, 8]:
                test_psnrs(self.logger_func, B, cur_out_log_folder, model_ckpt_name,
                           test_dir=test_dir, model_comp_type=self.model.compression)
            self.logger_func("Evaluating PSNRs done")

            if early_stopping.early_stop:
                self.logger_func('*** Early stopping ***')
                break

        self.logger_func("*** Training Completed ***")
        return self.model

    def test_model(self, dataloaders):
        self.logger_func("*" * 25)
        self.logger_func(f">> Testing {self.model_name} network")
        epoch_loss = self.test_epoch(dataloaders['test'])
        self.logger_func("Total test loss:", epoch_loss)

    def predict_transformed_image(self, img):
        img_transformed = img.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_transformed[None, :])[0]
        n = outputs.permute(1, 2, 0).detach().cpu().numpy() * 255
        image = Image.fromarray(n.astype(np.uint8), mode='RGB')
        return image

    def predict_image(self, img_path):
        test_data_transforms = T.Compose([
            # T.Resize((224, 224)),
            T.ToTensor(),
        ])
        img = Image.open(img_path).convert('RGB')
        img_transformed = test_data_transforms(img).to(self.device)
        return self.predict_transformed_image(img_transformed)

    def val_model(self, image_dir, out_dir, cnt=None):
        os.makedirs(out_dir, exist_ok=True)
        cur_cnt = 0
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(root, file)
                    if cnt is not None and cur_cnt >= cnt:
                        return
                    pil_img = self.predict_image(img_path)
                    pil_img.save(f'{out_dir}/{Path(img_path).stem}.png')
                    cur_cnt += 1
