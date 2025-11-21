import argparse
import os
import yaml
import torch
import numpy as np
from dataset_eval import Data
from models import DenoisingDiffusion, DiffusiveRestoration
import torch.distributed as dist
import time  # 添加这行
os.environ['NCCL_P2P_DISABLE']='1' 


def config_get():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='configs.yml', type=str, required=False, help="Path to the config file")
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    config = config_get()
    args = parser.parse_args()

    # 判断是否使用 cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=> using device: {}".format(device))
    config.device = device

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    # 判断是否使用 cuda
    config.local_rank = args.gpu
    device = torch.device("cuda", config.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    run = 0
    print(f"\n=== Starting Run {run + 1}/90 ===")
    
    # 使用时间戳生成新的种子
    current_seed = int(time.time()) + run  # 每次增加一个偏移量，保证种子不同
    print(f"Using seed for run {run + 1}: {current_seed}")  
    
    # 使用新的种子替换配置中的种子
    config.training.seed = current_seed

    # 设置随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    DATASET = Data(config)
    val_loader = DATASET.get_loaders(parse_patches=False, test=True)

    # 创建模型
    print(f"=> creating diffusion model for run {run + 1}")
    diffusion = DenoisingDiffusion(config, test=True)
    model = DiffusiveRestoration(diffusion, config)

    # 修改保存路径
    original_save_dir = config.data.test_save_dir
    config.data.test_save_dir = os.path.join(original_save_dir, f'run_{run + 1}')
    print(config.data.test_save_dir)
    os.makedirs(config.data.test_save_dir, exist_ok=True)

    # 恢复图像
    model.restore(val_loader, r=config.data.grid_r)

    # 记录使用的种子
    if dist.get_rank() == 0:  # 只在主进程中记录
        log_dir = os.path.dirname(original_save_dir)
        log_file = os.path.join(log_dir, "seed_log.txt")
        with open(log_file, "a") as f:
            f.write(f"Run {run + 1} at {time.strftime('%Y-%m-%d %H:%M:%S')}, seed: {current_seed}\n")

    # 恢复原始保存路径
    config.data.test_save_dir = original_save_dir

    # 等待所有进程完成当前运行
    dist.barrier()
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        


if __name__ == '__main__':
    main()
