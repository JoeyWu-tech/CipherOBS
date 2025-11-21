import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.ids_encoder import IDSEncoder
import time,re

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform


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

class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)


        self.input_dict = extract_chinese_characters(f"{self.root}/{self.phase}/run_1")
        self.target_dict = extract_chinese_characters(f"{self.root}/{self.phase}/TargetImage/style0")
        self.chinese_list = list(self.input_dict.keys())
        print()

        ids_path = '/home/hdd2/zhanggangjian/diff-han/han_ids.txt'
        glyph_path = '/home/hdd2/zhanggangjian/diff-han/glyphs.json'
        self.ids_encoder = IDSEncoder(ids_path, glyph_path, 32)
        print("")
        
    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                if '\ue83b' in img:
                    print(img_path)
                    continue
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style
        # print("")
        # self.target_images.pop(21045)
        # print("")

    def __getitem__(self, index):
        current_time_seed = int(time.time()*10+index)
        random.seed(current_time_seed)
        random_number = random.randint(0, len(self.input_dict)-1)
        chinese = self.chinese_list[random_number]
        a = self.input_dict[chinese]
        b = self.target_dict[chinese]
        content_image_path = random.sample(a, 1)[0]#.split('/')[-1]
        target_image_path = random.sample(b, 1)[0]#.split('/')[-1]




        # target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        # style, content = target_image_name.split('.')[0].split('+')
        style, content = 'style0', target_image_name.split('.')[0]#.split('+')

        # Read content image
        # content_image_path = f"{self.root}/{self.phase}/run_1/{content}.png"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        # images_related_style = self.style_to_images[style].copy()
        # images_related_style.remove(target_image_path)
        style_image_path = "/home/hdd2/zhanggangjian/StageTwo_Diffusion/dataset/style0+test_3.png"#random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        



        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            'fantizi':content.split('_')[1]}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
