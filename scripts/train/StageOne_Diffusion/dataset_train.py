import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from torch.utils.data.distributed import DistributedSampler
import time


class Data:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, test=False):

        train_path = os.path.join(self.config.data.train_data_dir)
        val_path = os.path.join(self.config.data.test_data_dir)

        train_dataset = MyDataset(train_path,
                                  n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  keep_image_size=self.config.data.training_keep_image_size,
                                  transforms=self.transforms,
                                  parse_patches=parse_patches)
        val_dataset = MyDataset(val_path,
                                n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                keep_image_size=self.config.data.testing_keep_image_size,
                                transforms=self.transforms,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # 训练数据

        # 评估数据
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
        #                                          shuffle=True, num_workers=self.config.data.num_workers,
        #                                          pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True, sampler=DistributedSampler(val_dataset))

        if not test:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                       shuffle=False, sampler=DistributedSampler(train_dataset),
                                                       num_workers=self.config.data.num_workers,
                                                       prefetch_factor=2,
                                                       pin_memory=True)
            return train_loader, val_loader

        if test:
            return val_loader


def extract_chinese_characters(directory_path):
    # 字典用于存储汉字对应的文件路径列表
    chinese_dict = {}

    # 遍历目录中的文件
    for filename in os.listdir(directory_path):
        # 使用正则表达式提取文件名中的汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', filename)
        
        # 获取文件的绝对路径
        file_path = os.path.join(directory_path, filename)

        # 将文件路径添加到汉字对应的列表中
        for char in chinese_chars:
            if char not in chinese_dict:
                chinese_dict[char] = []
            chinese_dict[char].append(file_path)

    return chinese_dict



# 数据集加载类
class MyDataset(torch.utils.data.Dataset):
    parse_patches: bool

    def __init__(self, dir, patch_size, n, keep_image_size, transforms, parse_patches=True):
        super().__init__()

        self.dir = dir
        input_names = os.listdir(dir + 'input')
        gt_names = os.listdir(dir + 'target')

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.keep_image_size = keep_image_size


        self.input_dict = extract_chinese_characters(dir + 'input')
        self.target_dict = extract_chinese_characters(dir + 'target')
        self.chinese_list = list(self.input_dict.keys())
        print()



    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):

        current_time_seed = int(time.time()*10+index)
        random.seed(current_time_seed)
        random_number = random.randint(0, len(self.input_dict)-1)
        chinese = self.chinese_list[random_number]
        a = self.input_dict[chinese]
        b = self.target_dict[chinese]
        input_name = random.sample(a, 1)[0].split('/')[-1]
        gt_name = random.sample(b, 1)[0].split('/')[-1]


        # input_name = self.input_names[index]
        # gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, 'input', input_name)).convert(
            'RGB') if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)).convert(
                'RGB') if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')
        
        if not self.keep_image_size:
            input_img = input_img.resize((100, 100), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((100, 100), PIL.Image.LANCZOS)
        else:
            wd_new, ht_new = input_img.size
            
            if wd_new < self.patch_size or ht_new < self.patch_size:
                ratio = max(self.patch_size / wd_new, self.patch_size / ht_new)
                wd_new = int(wd_new * ratio)
                ht_new = int(ht_new * ratio)
            
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
