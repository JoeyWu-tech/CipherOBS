from FontDiffuser.sample import arg_parse, sampling, load_fontdiffuer_pipeline
import PIL.Image as Image
import os
import torch
import numpy as np

# 初始化参数
Font_args = arg_parse()
Font_args.demo = True
Font_args.ckpt_dir = 'FontDiffuser/ckpt_r'
pipe = load_fontdiffuer_pipeline(args=Font_args)

def convert_to_rgb(image):
    """将灰度图像转换为RGB图像,如果已经是RGB,则直接返回"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def run_fontdiffuer(source_image,
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    seed):
    Font_args.character_input = False if source_image is not None else True
    Font_args.sampling_step = sampling_step
    Font_args.guidance_scale = guidance_scale
    Font_args.seed = seed

    # 转换为RGB格式
    source_image = convert_to_rgb(source_image)
    reference_image = convert_to_rgb(reference_image)

    # 执行风格迁移
    out_image = sampling(
        args=Font_args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image)

    # 调整输出大小
    desired_size = (112, 112)
    out_image = out_image.resize(desired_size)
    return out_image

# 读取待 refine 的图片路径
refine_path = '../data/modern_kanji'
refine_files = [f for f in os.listdir(refine_path) if f.endswith('.png')]

# 设置随机种子
seed = 61
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 获取所有参考图像路径
ref_image_dir = 'FontDiffuser/figures/ref_imgs'
ref_image_files = [f for f in os.listdir(ref_image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 遍历每张参考图像并生成对应风格的输出
for idx, ref_file in enumerate(ref_image_files, start=6):
    ref_image_path = os.path.join(ref_image_dir, ref_file)
    reference_image = Image.open(ref_image_path)

    # 输出路径：style0, style1, ...
    refine_result_path = f'../data/output/style{idx}'
    if not os.path.exists(refine_result_path):
        os.makedirs(refine_result_path)

    print(f"Processing style{idx} using reference image: {ref_file}")
    for file in refine_files:
        source_image = Image.open(os.path.join(refine_path, file))
        out_image = run_fontdiffuer(source_image, reference_image, 20, 7.5, seed)
        out_image.save(os.path.join(refine_result_path, file))
        print(f'{file} refined with style{idx}')
