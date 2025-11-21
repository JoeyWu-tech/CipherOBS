import os
import subprocess
import time
import sys
from datetime import datetime

def run_evaluation(total_runs=90, gpus="1,2,3"):
    base_save_dir = "After/result_90"  # 基础保存目录
    
    # 确保基础目录存在
    os.makedirs(base_save_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(base_save_dir, "evaluation_log.txt")
    
    def log_message(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

    # 计算使用的GPU数量
    num_gpus = len(gpus.split(','))
    log_message(f"使用GPU: {gpus} (共{num_gpus}个GPU)")

    # 设置环境变量
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = gpus

    # 运行评估
    for run in range(total_runs):
        run_dir = os.path.join(base_save_dir, f"run_{run + 1}")  # 从run_15开始
        os.makedirs(run_dir, exist_ok=True)
        
        log_message(f"开始运行 Run {run + 1}/80 (保存到 run_{run + 1})")
        """
        try:
            # 使用 torchrun 运行分布式评估
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",  # 使用指定数量的GPU
                "--master_port=29500",  # 指定主端口
                "eval_diffusion.py"
            ]
            
            # 运行命令，传入环境变量
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=my_env  # 使用修改后的环境变量
            )
            
            # 实时输出日志
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    log_message(output.strip())
            
            # 获取返回码
            return_code = process.poll()
            
            if return_code == 0:
                log_message(f"Run {run + 1} 成功完成")
            else:
                log_message(f"Run {run + 1} 失败，返回码: {return_code}")
            
        except Exception as e:
            log_message(f"运行出错: {str(e)}")
            continue
        
        # 每次运行之间暂停一小段时间
        time.sleep(5)
        """
    log_message("所有评估运行完成")

if __name__ == '__main__':
    # 可以在命令行参数中指定GPU
    if len(sys.argv) > 1:
        gpus = sys.argv[1]
    else:
        gpus = "1,2,3"  # 默认使用0-3号GPU
    
    try:
        run_evaluation(gpus=gpus)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)