from sample import arg_parse, sampling, load_fontdiffuer_pipeline
import PIL.Image as Image
import os
import torch
import numpy as np


import argparse

# parser = argparse.ArgumentParser(description="Process some paths.")

# # 添加参数
# parser.add_argument('--ckpt_dir', type=str, required=True, 
#                     help='Path to the checkpoint directory.')
# parser.add_argument('--refine_inter_result_path', type=str, required=True, 
#                     help='Path to the refined intermediate result.')

# 解析参数
# args = parser.parse_args()

Font_args = arg_parse()
Font_args.demo = True
Font_args.ckpt_dir = '/home/hdd2/zhanggangjian/StageTwo_Diffusion/outputs/sim2obc_ids_new_data_0623/global_step_50000'
refine_result_path = '/home/hdd2/zhanggangjian/new_data/test/input_result_0608_aug/run_1_wids_stage2infer_50000_vis'                      #After/result_refined_90/run_1_24

pipe = load_fontdiffuer_pipeline(args=Font_args)


def run_fontdiffuer(source_image,
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    seed):
    Font_args.character_input = False if source_image is not None else True
    Font_args.sampling_step = sampling_step
    Font_args.guidance_scale = guidance_scale
    Font_args.seed = seed
    if reference_image is None:
        reference_image = Image.open('/home/hdd2/zhanggangjian/StageTwo_Diffusion/dataset/style0+test_3.png')
    out_image, inter_images = sampling(
        args=Font_args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image)
    desired_size = (112, 112)
    out_image = out_image.resize(desired_size)

    inter_images = [image[0].resize(desired_size) for image in inter_images]

    return out_image, inter_images


# 读取refine_path下所有图片
refine_path = '/home/hdd2/zhanggangjian/new_data/test/input_result_0608_aug/run_1'                          #'After/result_90/run_1
if not os.path.exists(refine_result_path):
    os.makedirs(refine_result_path)

# if not os.path.exists(refine_inter_result_path):
#     os.makedirs(refine_inter_result_path)


refine_files = os.listdir(refine_path)
seed = 61
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
for file in refine_files:
    if file.endswith('.png'):
        source_image = Image.open(os.path.join(refine_path, file))
        out_image, inter_images = run_fontdiffuer(source_image, None, 20, 7.5, seed)
        out_image.save(os.path.join(refine_result_path, file))

        # dir_path = os.path.join(refine_inter_result_path, file)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for idx, images in enumerate(inter_images) :
        #     images.save(os.path.join(dir_path, str(idx)+'.png'))

        print(f'{file} refined')
        print(f"Saving to: {os.path.abspath(os.path.join(refine_result_path, file))}")

