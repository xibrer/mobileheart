import os

import numpy as np
# from thop import profile
import scipy.io as io
import torch
from torch.utils.data import Dataset, random_split


class MyDataset(Dataset):
    def __init__(self, root, user, train):
        self.mat_list = []
        self.noise_list = []
        self.data_length = 640
        self.user = user
        self.noise_dir = os.path.abspath('./datasets/noise/')
        self.train = train
        # merge all files in folder to a list
        if train:
            for user_folder in os.listdir(root):
                file_path = os.path.join(root, user_folder)
                if user_folder not in [user]:
                    for file in os.listdir(file_path):
                        mat_file = os.path.join(file_path, file)
                        self.mat_list.append(mat_file)
        else:
            for user_folder in os.listdir(root):
                file_path = os.path.join(root, user_folder)
                if user_folder in [user]:
                    for file in os.listdir(file_path):
                        mat_file = os.path.join(file_path, file)
                        self.mat_list.append(mat_file)

        for user_folder in os.listdir(self.noise_dir):
            noise_file_path = os.path.join(self.noise_dir, user_folder)
            for noise_file in os.listdir(noise_file_path):
                full_path = os.path.join(noise_file_path, noise_file)
                self.noise_list.append(full_path)

    def __getitem__(self, index):
        data = io.loadmat(self.mat_list[index])
        j = np.random.randint(0, len(self.noise_list))
        noise = io.loadmat(self.noise_list[j])
        mask_p = 0.01
        mask = np.random.choice([True, False], size=self.data_length, p=[1 - mask_p, mask_p])
        self.x = torch.from_numpy(data['y'].astype(np.float32))
        self.noise = torch.from_numpy(noise['y'].astype(np.float32))
        if self.train:
            x_i = self.x + self.noise
            x_i = (x_i - x_i.min()) / (x_i.max() - x_i.min())
            # 如果mask中的元素为True，则在相应的masked_data位置选择x_i数组中的元素
            # 如果mask中的元素为False，则在相应的masked_data位置选择0
            x_i = np.where(mask, x_i, 0)
        else:
            x_i = self.x
            x_i = (x_i - x_i.min()) / (x_i.max() - x_i.min())
        y = (self.x - self.x.min()) / (self.x.max() - self.x.min())

        return x_i, y

    def __len__(self):
        return len(self.mat_list)

    def split(self, val_ratio):
        """
        划分训练集和验证集的函数，接收一个参数：`val_ratio`（验证集的比例）。
        这个函数会返回划分好的训练集和验证集。
        """
        val_size = int(val_ratio * len(self))
        train_size = len(self) - val_size
        train_set, val_set = random_split(self, [train_size, val_size])

        return train_set, val_set
