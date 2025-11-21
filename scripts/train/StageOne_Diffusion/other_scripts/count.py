import os
import json
import random
import shutil


# 文件夹路径设置
test_input_folder = "OwnData/test/input"  
test_target_folder = "MyData/test/target"  
train_input_folder = "MyData/train/input"  
train_target_folder = "MyData/train/target" 
test_rusult = "OwnData/result"
test_rusult_r = "trainData/result_refined"

image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']  # 常见图片格式
image_count_test_input = sum(1 for file in os.listdir(test_input_folder) 
                  if os.path.isfile(os.path.join(test_input_folder, file)) and os.path.splitext(file)[1].lower() in image_extensions)

image_count_test_target = sum(1 for file in os.listdir(test_target_folder) 
                  if os.path.isfile(os.path.join(test_target_folder, file)) and os.path.splitext(file)[1].lower() in image_extensions)

image_count_train_input = sum(1 for file in os.listdir(train_input_folder) 
                  if os.path.isfile(os.path.join(train_input_folder, file)) and os.path.splitext(file)[1].lower() in image_extensions)

image_count_train_target = sum(1 for file in os.listdir(train_target_folder) 
                  if os.path.isfile(os.path.join(train_target_folder, file)) and os.path.splitext(file)[1].lower() in image_extensions)

image_count_test_result = sum(1 for file in os.listdir(test_rusult) 
                  if os.path.isfile(os.path.join(test_rusult, file)) and os.path.splitext(file)[1].lower() in image_extensions)

image_count_test_result = sum(1 for file in os.listdir(test_rusult_r) 
                  if os.path.isfile(os.path.join(test_rusult, file)) and os.path.splitext(file)[1].lower() in image_extensions)

print(f"test_input 文件夹中共有 {image_count_test_input} 张图片")
print(f"test_target 文件夹中共有 {image_count_test_target} 张图片")
print(f"train_input 文件夹中共有 {image_count_train_input} 张图片")
print(f"train_target 文件夹中共有 {image_count_train_target} 张图片")
print(f"test_result 文件夹中共有 {image_count_test_result} 张图片")