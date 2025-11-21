from paddleocr import PaddleOCR
import os

def calculate_top_k_accuracy(pred_texts, gt_text, k=3):
    """计算 Top-k 准确率"""
    # 将预测文本列表截取到前k个
    pred_texts = pred_texts[:k]
    # 如果真实标签在前k个预测中，返回1，否则返回0
    return 1 if gt_text.strip() in [pred.strip() for pred in pred_texts] else 0

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="ch",
    use_gpu=True,  # 使用 GPU
    gpu_mem=1000,
    det_limit_side_len=640,
    rec_batch_num=1,
    enable_mkldnn=False,
    cpu_threads=2,
    ir_optim=False,  # 禁用 IR 优化
    use_tensorrt=False,
    precision='fp32',  # 使用 FP32 精度
    drop_score=0.1
)

# 设置路径
img_folder = 'OBSD/OBS_Diffusion/result_refined'  # 图片文件夹
gt_file = 'test_characters.txt'     # 真实标签文件
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# 读取真实标签
gt_dict = {}
with open(gt_file, 'r', encoding='utf-8') as f:
    for line in f:
        img_name, label = line.strip().split('\t')  # 根据实际分隔符调整
        gt_dict[img_name] = label

# 计算准确率
total = 0
correct = 0

with open('accuracy_results_Top20.txt', 'w', encoding='utf-8') as f:
    for filename in os.listdir(img_folder):
        if filename.lower().endswith(img_extensions):
            img_path = os.path.join(img_folder, filename)
            
            # OCR识别
            result = ocr.ocr(img_path, cls=True)
            if result[0]:
                # 获取所有识别结果，按置信度排序
                pred_texts = [line[1][0] for line in result[0]]
                
                # 获取对应的真实标签
                gt_text = gt_dict.get(filename)
                if gt_text is not None:
                    total += 1
                    is_correct = calculate_top_k_accuracy(pred_texts, gt_text, k=20)
                    correct += is_correct
                    
                    # 记录每个样本的识别结果
                    f.write(f"Image: {filename}\n")
                    f.write(f"Top-20 Predictions:\n")
                    for i, pred in enumerate(pred_texts[:20], 1):
                        f.write(f"{i}. {pred}\n")
                    f.write(f"Ground Truth: {gt_text}\n")
                    f.write(f"In Top-20: {is_correct}\n\n")

    # 计算总体准确率
    accuracy = correct / total if total > 0 else 0
    f.write(f"\nTop-20 Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Top-20 Accuracy: {accuracy:.4f} ({correct}/{total})")