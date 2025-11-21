from paddleocr import PaddleOCR
import os

def calculate_accuracy(pred_text, gt_text):
    """计算单个样本的准确率"""
    return 1 if pred_text.strip() == gt_text.strip() else 0

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="ch",
    use_gpu=False,  # 使用 GPU
    gpu_mem=1000,
    det_limit_side_len=640,
    rec_batch_num=1,
    enable_mkldnn=False,
    cpu_threads=2,
    ir_optim=False,  # 禁用 IR 优化
    use_tensorrt=False,
    precision='fp32'  # 使用 FP32 精度
)

# 设置路径
img_folder = 'OwnData/result_refined'  # 图片文件夹
gt_file = 'OwnData/test/ground_truth.txt'     # 真实标签文件
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# 读取真实标签（假设格式为：图片名 标签文本）
gt_dict = {}
with open(gt_file, 'r', encoding='utf-8') as f:
    for line in f:
        img_name, label = line.strip().split('\t')  # 根据实际分隔符调整
        gt_dict[img_name] = label

# 计算准确率
total = 0
correct = 0

with open('OwnData/accuracy_results.txt', 'w', encoding='utf-8') as f:
    for filename in os.listdir(img_folder):
        if filename.lower().endswith(img_extensions):
            img_path = os.path.join(img_folder, filename)
            
            # OCR识别
            result = ocr.ocr(img_path, cls=True)
            if result[0]:  # 确保有识别结果
                pred_text = result[0][0][1][0]  # 获取第一个（置信度最高的）识别结果
                
                # 获取对应的真实标签
                gt_text = gt_dict.get(filename)
                if gt_text is not None:
                    total += 1
                    is_correct = calculate_accuracy(pred_text, gt_text)
                    correct += is_correct
                    
                    # 记录每个样本的识别结果
                    f.write(f"Image: {filename}\n")
                    f.write(f"Predicted: {pred_text}\n")
                    f.write(f"Ground Truth: {gt_text}\n")
                    f.write(f"Correct: {is_correct}\n\n")

    # 计算总体准确率
    accuracy = correct / total if total > 0 else 0
    f.write(f"\nTop-1 Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Top-1 Accuracy: {accuracy:.4f} ({correct}/{total})")