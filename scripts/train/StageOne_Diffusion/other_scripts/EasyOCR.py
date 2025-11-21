import easyocr

def get_top_n_single_image(image_path, n=20):
    """获取单张图片的 Top-N 识别结果"""
    # 初始化 reader
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    
    # 使用 beamsearch 获取多个候选结果
    results = reader.readtext(
        image_path,
        decoder='beamsearch',  # 使用 beam search 解码
        beamWidth=n,          # beam width 设为 N
        batch_size=1,
        detail=1              # 返回详细信息
    )
    
    # 按置信度排序
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    top_n = sorted_results[:n]
    
    # 打印结果
    print(f"\n图片路径: {image_path}")
    print("\nTop-N 识别结果:")
    for idx, (bbox, text, conf) in enumerate(top_n, 1):
        print(f"Top {idx}: {text} (置信度: {conf:.4f})")
    
    return top_n

# 使用示例
image_path = 'OBSD/OBS_Diffusion/result_refined/test_1.png'  # 替换为你的图片路径
results = get_top_n_single_image(image_path, n=20)
print(results)