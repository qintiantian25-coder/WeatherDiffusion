import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class AllWeather:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='snow'):
        # if validation == 'raindrop':
        #     print("=> evaluating raindrop test set...")
        #     path = os.path.join(self.config.data.data_dir, 'data', 'raindrop', 'test')
        #     filename = 'raindroptesta.txt'
        # elif validation == 'rainfog':
        #     print("=> evaluating outdoor rain-fog test set...")
        #     path = os.path.join(self.config.data.data_dir, 'data', 'outdoor-rain')
        #     filename = 'test1.txt'
        # else:   # snow
        #     print("=> evaluating snowtest100K-L...")
        #     path = os.path.join(self.config.data.data_dir, 'data', 'snow100k')
        #     filename = 'snowtest100k_L.txt'

        # train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, 'data', 'allweather'),
        #                                   n=self.config.training.patch_n,
        #                                   patch_size=self.config.data.image_size,
        #                                   transforms=self.transforms,
        #                                   filelist='allweather.txt',
        #                                   parse_patches=parse_patches)
        # val_dataset = AllWeatherDataset(path, n=self.config.training.patch_n,
        #                                 patch_size=self.config.data.image_size,
        #                                 transforms=self.transforms,
        #                                 filelist=filename,
        #                                 parse_patches=parse_patches)

        # 1. 训练集加载 (保持不变，使用 allweather.txt)
        train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, 'data', 'allweather'),
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          filelist='allweather.txt',
                                          parse_patches=parse_patches)

        # 2. 确定验证/测试集的路径和文件列表

        if parse_patches or validation != 'custom_test':
            # 默认：用于训练时的验证集或旧的预设测试集
            validation_dir = os.path.join(self.config.data.data_dir, 'data', 'allweather')
            validation_filelist = 'validation.txt'
            print(f"=> 加载默认验证集列表:{validation_filelist}")

        elif not parse_patches and validation == 'custom_test':
            print("--------------------------------------------------")
            print("=> 检测到测试阶段，加载自定义测试集...")

            # 1. 找到当前脚本所在目录 (datasets/)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 2. 向上退一级到项目根目录 (..)
            project_root = os.path.dirname(current_dir)

            # 3. 构造新的绝对路径：从项目根目录进入 data/custom_data/test/image
            # 目标路径: [Project_Root] / data / custom_data / test / image
            validation_dir = os.path.join(project_root, 'data', 'custom_data', 'test', 'image')
            validation_filelist = 'test_list.txt'

            print(f"✅ 启用自定义测试集路径: {validation_dir}")
            print(f"✅ 使用文件列表: {validation_filelist}")
            print("--------------------------------------------------")
        else:
            # 确保所有路径都有默认处理，例如兼容原有的 snow/raindrop 逻辑
            # 这里使用 snow 作为 fallback，但这取决于您项目中被注释掉的代码
            validation_dir = os.path.join(self.config.data.data_dir, 'data', 'snow100k')
            validation_filelist = 'snowtest100k_L.txt'

        # 3. 统一实例化 val_dataset (只执行一次)
        val_dataset = AllWeatherDataset(validation_dir,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=validation_filelist,
                                        parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        self.dir = dir
        train_list = os.path.join(dir, filelist)
        with open(train_list) as f:
            contents = f.readlines()

            # 【关键修改】：读取 Input 和 GT 两列
            input_names = [i.strip().split(' ')[0] for i in contents]
            gt_names = [i.strip().split(' ')[1] for i in contents]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

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
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

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
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
