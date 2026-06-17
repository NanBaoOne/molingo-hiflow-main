# MoLingo / HiFlow-SAE 二阶段训练启动教程

本文面向已将项目上传到远端服务器、准备直接训练二阶段模型的用户。二阶段训练依赖已经训练好的 SAE 模型；本流程只运行 MoLingo / HiFlow-SAE 的二阶段训练，不需要运行 SAE 训练代码。

## 1. 项目目录说明

服务器上的实际运行目录固定为：

```bash
/mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow
```

后续所有命令都建议在该目录下执行：

```bash
cd /mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow
```

常用目录说明：

```text
mogen-hiflow/
├── train_molingo.py      # 二阶段训练入口，MoLingo 和 HiFlow-SAE 都使用该脚本
├── eval_mogen.py         # 独立评估入口
├── train_sae.py          # SAE 训练入口，本教程不需要运行
├── checkpoints/          # SAE 权重和二阶段训练输出目录
├── models/               # 模型代码
├── mogen/                # mogen 包代码
├── options/              # 训练和评估参数
└── data/                 # 数据相关文件
```

## 2. SAE 模型放置目录格式

二阶段训练前，需要先把已经训练好的 SAE 模型放到 `checkpoints/ms/<sae_name>/` 下。

目录格式必须包含：

```text
checkpoints/ms/<sae_name>/opt.txt
checkpoints/ms/<sae_name>/model/net_best_fid.ckpt
```

示例：

```text
checkpoints/ms/ms_large_vae/opt.txt
checkpoints/ms/ms_large_vae/model/net_best_fid.ckpt
```

其中 `<sae_name>` 是 SAE 实验名称，训练二阶段时需要通过 `--sae_name` 指定。

## 3. 只训练二阶段

本教程只训练二阶段模型：

```text
原始 MoLingo 二阶段
HiFlow-SAE 二阶段
```

不需要运行下面的 SAE 训练脚本：

```bash
python train_sae.py
```

只要 SAE 的 `opt.txt` 和 checkpoint 已经按目录格式放好，就可以直接启动二阶段训练。

## 4. 环境准备

进入实际运行目录：

```bash
cd /mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow
```

激活 Python / Conda 环境。环境名请按服务器实际情况替换：

```bash
conda activate your_env_name
```

确认 PyTorch 可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

确认 `transformers` 可用：

```bash
python -c "import transformers; print(transformers.__version__)"
```

确认 T5-large 可访问或已经缓存。二阶段训练会用到 T5-large 文本编码器，如果服务器不能联网，需要提前把模型缓存好，或者配置本地缓存路径。

可用下面的命令简单检查：

```bash
python -c "from transformers import T5Tokenizer, T5EncoderModel; T5Tokenizer.from_pretrained('t5-large'); T5EncoderModel.from_pretrained('t5-large'); print('t5-large ready')"
```

## 5. 原始 MoLingo 二阶段训练命令

将 `<sae_name>` 替换为实际 SAE 名称，例如 `ms_large_vae`。

```bash
python train_molingo.py \
  --dataset_name ms \
  --name molingo_ms_large_vae_<sae_name> \
  --second_stage molingo \
  --checkpoints_dir checkpoints \
  --sae_name <sae_name>
```

如果使用非分布式单卡启动，直接运行上面的 `python train_molingo.py ...` 即可。

## 6. HiFlow-SAE 二阶段训练命令

将 `<sae_name>` 替换为实际 SAE 名称，例如 `ms_large_vae`。

```bash
python train_molingo.py \
  --dataset_name ms \
  --name hiflow_ms_large_vae_<sae_name> \
  --second_stage hiflow \
  --checkpoints_dir checkpoints \
  --sae_name <sae_name> \
  --hiflow_scales 0.3,0.6,1.0 \
  --time_patch 2
```

该命令会读取：

```text
checkpoints/ms/<sae_name>/opt.txt
checkpoints/ms/<sae_name>/model/net_best_fid.ckpt
```

## 7. SAE checkpoint 不是默认名称

默认 SAE checkpoint 名称为：

```text
net_best_fid.ckpt
```

如果你的 SAE 权重文件不是默认名称，例如：

```text
checkpoints/ms/<sae_name>/model/net_last.ckpt
```

启动二阶段训练时添加 `--sae_ckpt_name`：

```bash
python train_molingo.py \
  --dataset_name ms \
  --name hiflow_ms_large_vae_<sae_name> \
  --second_stage hiflow \
  --checkpoints_dir checkpoints \
  --sae_name <sae_name> \
  --sae_ckpt_name net_last.ckpt \
  --hiflow_scales 0.3,0.6,1.0 \
  --time_patch 2
```

原始 MoLingo 二阶段同样可以添加：

```bash
python train_molingo.py \
  --dataset_name ms \
  --name molingo_ms_large_vae_<sae_name> \
  --second_stage molingo \
  --checkpoints_dir checkpoints \
  --sae_name <sae_name> \
  --sae_ckpt_name net_last.ckpt
```

## 8. 训练输出目录说明

训练输出会写入 `--checkpoints_dir` 指定的目录下，即：

```text
checkpoints/ms/<run_name>/
```

HiFlow-SAE 推荐 run 名称：

```text
hiflow_ms_large_vae_<sae_name>
```

对应输出目录：

```text
checkpoints/ms/hiflow_ms_large_vae_<sae_name>/
```

原始 MoLingo 推荐 run 名称：

```text
molingo_ms_large_vae_<sae_name>
```

对应输出目录：

```text
checkpoints/ms/molingo_ms_large_vae_<sae_name>/
```

二阶段训练得到的最佳 checkpoint 通常位于：

```text
checkpoints/ms/<run_name>/model/net_best_fid.pth
```

例如：

```text
checkpoints/ms/hiflow_ms_large_vae_ms_large_vae/model/net_best_fid.pth
checkpoints/ms/molingo_ms_large_vae_ms_large_vae/model/net_best_fid.pth
```

## 9. 独立评估命令

评估 HiFlow-SAE 二阶段模型：

```bash
python eval_mogen.py \
  --dataset_name ms \
  --dim_pose 272 \
  --model_dir checkpoints/ms/hiflow_ms_large_vae_<sae_name>/model/net_best_fid.pth
```

评估原始 MoLingo 二阶段模型：

```bash
python eval_mogen.py \
  --dataset_name ms \
  --dim_pose 272 \
  --model_dir checkpoints/ms/molingo_ms_large_vae_<sae_name>/model/net_best_fid.pth
```

如果评估脚本要求传入目录而不是 `.pth` 文件，请使用对应 run 目录或 `model` 目录，并以脚本实际报错为准：

```bash
python eval_mogen.py \
  --dataset_name ms \
  --dim_pose 272 \
  --model_dir checkpoints/ms/hiflow_ms_large_vae_<sae_name>
```

## 10. 常见问题排查

### import mogen 失败

必须在实际运行目录执行命令：

```bash
cd /mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow
```

如果仍然失败，显式设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow:$PYTHONPATH
```

再测试：

```bash
python -c "import mogen; print('import mogen ok')"
```

### 找不到 SAE

检查 `--sae_name` 是否和目录名完全一致：

```bash
ls checkpoints/ms/<sae_name>
ls checkpoints/ms/<sae_name>/model
```

必须存在：

```text
checkpoints/ms/<sae_name>/opt.txt
checkpoints/ms/<sae_name>/model/net_best_fid.ckpt
```

如果 ckpt 文件名不是 `net_best_fid.ckpt`，添加：

```bash
--sae_ckpt_name your_checkpoint_name.ckpt
```

### T5-large 下载失败

如果服务器不能联网，训练时可能在加载 `t5-large` 时报错。解决方式：

```text
1. 在可联网机器提前下载 t5-large。
2. 上传到服务器的 Hugging Face 缓存目录，或项目可访问的本地目录。
3. 确认 transformers 能从缓存加载。
```

检查命令：

```bash
python -c "from transformers import T5Tokenizer, T5EncoderModel; T5Tokenizer.from_pretrained('t5-large'); T5EncoderModel.from_pretrained('t5-large'); print('t5-large ready')"
```

### CUDA / 显存问题

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

查看显卡占用：

```bash
nvidia-smi
```

如果显存不足，可以优先尝试：

```text
1. 确认没有其他进程占用 GPU。
2. 使用更空闲的 GPU，例如设置 CUDA_VISIBLE_DEVICES。
3. 降低 batch size，具体参数名以 train_molingo.py / options 中支持的参数为准。
```

单卡示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train_molingo.py \
  --dataset_name ms \
  --name hiflow_ms_large_vae_<sae_name> \
  --second_stage hiflow \
  --checkpoints_dir checkpoints \
  --sae_name <sae_name> \
  --hiflow_scales 0.3,0.6,1.0 \
  --time_patch 2
```

### 非分布式启动

本教程命令默认是非分布式启动，不需要 `torchrun`：

```bash
python train_molingo.py ...
```

如果服务器或脚本没有特别要求多卡分布式，优先使用非分布式单卡命令，便于定位问题。

### 路径绝对值不要改

实际运行目录请保持为：

```bash
/mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow
```

不要把命令中的运行目录改成其他路径，除非项目实际上传位置已经改变。路径不一致会导致 `import mogen`、数据路径、checkpoint 路径或 T5 缓存加载失败。

## 11. 推荐启动顺序

先确认环境和 SAE 文件：

```bash
cd /mnt/user/lh3/MotionStreamer-main/molingo-hiflow/mogen-hiflow
conda activate your_env_name
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import mogen; print('import mogen ok')"
ls checkpoints/ms/<sae_name>/opt.txt
ls checkpoints/ms/<sae_name>/model/net_best_fid.ckpt
```

再启动 HiFlow-SAE 二阶段训练：

```bash
python train_molingo.py \
  --dataset_name ms \
  --name hiflow_ms_large_vae_<sae_name> \
  --second_stage hiflow \
  --checkpoints_dir checkpoints \
  --sae_name <sae_name> \
  --hiflow_scales 0.3,0.6,1.0 \
  --time_patch 2
```

训练完成后评估：

```bash
python eval_mogen.py \
  --dataset_name ms \
  --dim_pose 272 \
  --model_dir checkpoints/ms/hiflow_ms_large_vae_<sae_name>/model/net_best_fid.pth
```
