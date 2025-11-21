这份压缩包里的文件是第二次diffusion的脚本，也就是refinement部分的代码
    configs 
    dataset 
    figures 
    scripts 
    src 
    gradio_app.py 
    requirements.txt 
    sample.py 
    train.py 
    utils.py
    是由github库https://github.com/yeungchenwa/FontDiffuser中的原文件

    data_examples是由我们创建的数据集，其中data_examples/train/ContentImage里是我们自己生成的甲骨文图片，TargetImage是对应的原来的甲骨文图片

    开始训练是由scripts里 'sh train_phase_1.sh' 完成