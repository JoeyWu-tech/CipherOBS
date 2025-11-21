import os
import glob

def delete_extra_images(folder_path, target_count=5786):
    """删除文件夹中超出目标数量的图片"""
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(folder_path, '*.*'))  # 匹配所有文件
    
    # 按文件名排序
    image_files.sort()
    
    # 计算需要删除的文件数量
    current_count = len(image_files)
    if current_count <= target_count:
        print(f"文件夹 {folder_path} 中的图片数量（{current_count}）小于或等于目标数量（{target_count}），无需删除。")
        return
    
    # 删除多余的文件
    files_to_delete = image_files[target_count:]
    deleted_count = 0
    
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"删除文件 {file_path} 时出错: {e}")
    
    print(f"文件夹 {folder_path} 中已删除 {deleted_count} 个文件，现有 {target_count} 个文件。")

def main():
    # 设置路径
    input_folder = "MyData/test/input"
    target_folder = "MyData/test/target"
    
    # 确保文件夹存在
    if not os.path.exists(input_folder) or not os.path.exists(target_folder):
        print("错误：输入或目标文件夹不存在！")
        return
    
    # 删除多余的图片
    print("开始处理 input 文件夹...")
    delete_extra_images(input_folder)
    
    print("\n开始处理 target 文件夹...")
    delete_extra_images(target_folder)

if __name__ == "__main__":
    # 添加确认提示
    print("此操作将删除多余的图片文件，确保每个文件夹只保留5786张图片。")
    confirm = input("是否继续？(y/n): ")
    if confirm.lower() == 'y':
        main()
    else:
        print("操作已取消")