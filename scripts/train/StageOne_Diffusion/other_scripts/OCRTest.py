from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(
    use_angle_cls=False,  # 关闭角度检测
    lang="ch",
    use_gpu=False,  # 明确使用 CPU
    det_limit_side_len=640,
    rec_batch_num=1,
    enable_mkldnn=False,  # 禁用 MKL-DNN
    cpu_threads=2,
    ir_optim=False,  # 禁用 IR 优化
    precision='fp32'  # 使用 FP32 精度
)  # need to run only once to download and load model into memory

font_path = 'FontDiffuser/ttf/KaiXinSongA.ttf'  # PaddleOCR 默认字体路径
if not os.path.exists(font_path):
    # 如果默认路径不存在，可以使用系统字体
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # Linux 系统字体
    # 或者
    # font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS 系统字体
    # font_path = 'C:/Windows/Fonts/simfang.ttf'  # Windows 系统字体

img_path = 'OBSD/OBS_Diffusion/result_refined/test_1.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# 显示结果
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')