# BABEL 语义监督 + MoMask，HumanML3D 272维

本目录提供独立实验：在原生 MoMask RVQ-VAE 编码器上增加局部 BABEL/T5
语义对齐，然后训练原生基础掩码 Transformer 与残差 Transformer。
**不是把现有连续 SAE 的 checkpoint 塞进 MoMask**：离散标记需要 RVQ。
已有兼容的 272维 RVQ 可以显式 warm start；服务器已有的数据/评估器直接复用。
263维权重和连续 SAE 权重不可无损替代 272维 RVQ。

## 代码来源与侵入范围

- `vendor/momask` 的 RVQ、MaskTransformer、ResidualTransformer 是官方文件原样复制；
  来源 https://github.com/EricGuo5513/momask-codes ，固定版本见 `UPSTREAM.json`，保留 MIT 许可证。
- 新增代码仅在本目录：数据适配、语义辅助损失、训练循环、TMR272 评估与启动脚本。
- 不修改 `mogen-hiflow`；通过运行时包路径适配复用其评估器类与指标实现。
- 局部语义监督借鉴 MoLingo / 本仓库 SAE：stride=4、历史窗口16帧、余弦对齐、1024维
  BABEL/T5 特征到运动特征的线性投影；在量化前编码特征上监督。
  本适配在每条序列内部按原始语义余弦去除重复位置，不跨样本过滤，不宣称逐位复刻旧 trainer。
- 272维显式位置重建项使用 `[8:74]`（22关节），不沿用263维的切片。
- 保留 RVQ 原生量化层数、量化 dropout、基础标记掩码策略和残差层预测机制。

## 服务器启动

在原来能运行项目的 CUDA 环境中，仓库根目录执行：

```bash
git pull --ff-only
python -m pip install -r 'sae+momask/requirements-extra.txt'
# 如果没有 OpenAI CLIP（import clip 应有 load/tokenize）：
# python -m pip install git+https://github.com/openai/CLIP.git
bash 'sae+momask/run.sh' check
bash 'sae+momask/run.sh' all
```

后台运行：

```bash
nohup bash 'sae+momask/run.sh' all > sae_momask272.log 2>&1 &
tail -f sae_momask272.log
```

服务器目录可以是 `/mnt/user/lh3/molingo-hiflow`，不依赖本机 Windows 路径。
脚本自行切换到实验目录，不需要重命名仓库、不需要 `mogen` 软链接。

默认输入路径（均可通过环境变量覆盖）：

```text
DATA_ROOT=/mnt/user/lh3/MotionStreamer-main/humanml3d_272
  motion_data/<id>.npy                [T,272]，30fps
  texts/<id>.txt                      caption#tokens#start_seconds#end_seconds
  babel_272_annotation_t5/<id>.npy     [T,1024]，与原运动逐帧对应
  mean_std/Mean.npy、Std.npy           [272]
  split/train.txt、val.txt、test.txt
EVALUATOR=/mnt/user/lh3/MotionStreamer-main/Evaluator_272/epoch=99.ckpt
EVAL_TEXT_MODEL=distilbert-base-uncased
```

CLIP `ViT-B/32` 和 DistilBERT 使用服务器已有缓存，缓存不存在时其原生加载器可能联网。
可以用 `EVAL_TEXT_MODEL=/服务器上的distilbert目录` 明确指定评估文本骨干。
不会自动扫描/替换你的权重，也不上传数据和大权重。

`check` 检查三个划分、形状与BABEL时间对应，并真实加载评估器进行一次嵌入前向；训练样本没有BABEL文件时仍参与重建，
仅跳过该样本语义项。有BABEL文件却时间错位会报错；全部无标注也会报错。
沿用原项目60–300帧源序列过滤、30fps文本子段及长度4倍数处理。
缺失运动/文本文件明确报错，长度过滤与标注覆盖率会输出报告。

## 46GB 单卡速度与进度

- 默认 `VQ_BATCH/MASK_BATCH/RES_BATCH=auto`。逐阶段隔离子进程，真实执行前向/反向/
  AdamW步骤测量显存和吞吐，从16到1024逐步探测，首次OOM后停止。
- 在总显存88%上限内选择实测吞吐最高的batch，给评估与外部进程留余量。
  它不保证恰好占满46GB；训练速度与安全余量优先于无意义地占满显存。
- 默认支持时用 BF16，否则 FP16 + GradScaler；RVQ量化/EMA与距离计算保持 FP32。
  支持TF32、多worker、pinned memory、持久worker；正式训练不启用anomaly detection。
- 交互终端显示 tqdm 进度、step、loss、学习率、显存峰值和预计剩余时间。
  文件日志只记录 `VALIDATION_LOSS` 与 `EVAL_METRICS`，不保存进度条和逐步训练输出。
  验证损失另存为 `validation_loss.csv`；最终评估仍有JSON/CSV结果文件。
  `.log` 默认每个20MiB、保留2个备份，可由 `LOG_MAX_MB/LOG_BACKUPS` 调整。
  异常诊断单独写入 `.error.log`；不会清理已有的旧日志。
  `nohup` 后台运行没有交互终端，自动关闭动态条；想持续看进度可在 tmux 中直接运行脚本。
  初始ETA不包括尚未测量的其他阶段和最终重复评估，运行一段时间后更稳定。
- 不自动按batch改变学习率；固定step预算下batch变化会改变样本曝光量，消融必须固定各阶段batch。
- 如其他任务占显存，先让出该GPU或显式设较小batch；自动探测不能预测外部任务后续显存变化。

例：

```bash
VQ_BATCH=512 MASK_BATCH=256 RES_BATCH=128 bash 'sae+momask/run.sh' all
# 仅为手动batch示例，不代表已在你的显卡验证，请优先使用auto。
```

默认预算为 VQ 50,000步、Mask 100,000步、Residual 100,000步，这是可配置的初始预算，
不代表保证收敛；通过 `VQ_STEPS/MASK_STEPS/RES_STEPS` 调整。
默认每2,000步保存并验证损失、每10,000步在val上评估TMR272，最终一步也评估。
`EVAL_EVERY=0` 只关闭中间指标评估，最终评估仍保留。
可分别 `bash run.sh vq/mask/res/eval` 启动；`all` 串行执行。

## 断点与权重

输出默认在本目录 `runs/lsa0.001_momask272_s3407/`，可用 `RUN_DIR` 指定绝对路径。
再次运行脚本会对存在 `latest.pt` 的阶段续训，并自动读取原batch。
保存模型、优化器、scaler、随机数状态以及官方RVQ未写入state_dict的EMA统计。
恢复不承诺多worker预取队列的逐样本位级复现；不会重新初始化已有EMA码本。

每阶段 `latest.pt`、验证损失最佳 `best.pt`、验证FID最佳 `best_fid.pt`；
后续阶段优先使用 `best_fid.pt`，没有时用 `best.pt`。
更换或继续训练Tokenizer后，必须重新训练匹配的Mask/Residual：不能沿用旧token标签。

```bash
# 使用已有兼容272维RVQ初始化；code_dim/width等必须与本实验配置相符。
INIT_VQ=/path/to/matching272rvq.tar bash 'sae+momask/run.sh' vq
# 精确指定本实验已训练的分阶段checkpoint：
VQ_CHECKPOINT=/path/to/vq/best_fid.pt bash 'sae+momask/run.sh' mask
```

不自动使用服务器上名字相似的权重，防止把263维/旧码本与本实验混用。

## 评估结果

最终在test上重复20次，输出JSON原始结果、汇总CSV、均值和95%正态近似置信区间：

- FID；R-Precision Top1/Top2/Top3；Matching Distance；Diversity；
- real-motion检索与Diversity参照、样本数、末批排除数、评估器与统计量SHA256。

与molingo-hiflow一致使用272维原始归一化输入、评估器分布的 `.loc`、不额外做嵌入L2归一化。
R-Precision候选池固定32、shuffle、drop_last；**训练batch不会改变检索池**。
本loader评估使用固定首条caption/确定性裁剪，旧项目loader有随机caption/crop；严格对表时，
应让两种方法共用同一批数据与评估种子，不能直接认定旧日志数值完全同协议。
置信区间来自固定权重的重复生成/评估，不能写成多个独立训练种子的置信区间。
只用val选权重，不用test调参或挑最好的采样设置。

## 两项建议对照：服务于“语义与层级”分析

1. 同预算、同batch训练 `LSA_WEIGHT=0` 与 `0.001` 两套独立RVQ+生成器，检验语义监督。
2. 固定训练好的模型，改变推理残差层数，分析层级细化的质量—计算关系：

```bash
NUM_RES_LAYERS=0 bash 'sae+momask/run.sh' eval  # 仅基础码，属于截断推理
NUM_RES_LAYERS=2 bash 'sae+momask/run.sh' eval  # 基础码+两层残差
NUM_RES_LAYERS=-1 bash 'sae+momask/run.sh' eval # 完整6层
```

各结果分别写入 `eval_test_res0.json`、`eval_test_res2.json`、`eval_test_res-1.json`。
截断推理不等于“单层模型重训练消融”，不能据此声称层级训练优于所有单层方法。
MoMask是残差量化层级，HiFlow是时间尺度层级；跨骨干比较不是只改流匹配/VQ-VAE一个变量。
本实验可支持不同层级生成路径下语义监督的适用性分析，不能独自证明新的统一层级理论。

第一轮auto探测后，查看每阶段 `batch_probe.json`，把选出的三个batch固定，
再用于LSA关闭组，确保相同step下曝光量可比：

```bash
LSA_WEIGHT=0 RUN_DIR=/path/to/baseline \
 VQ_BATCH=<vq选定值> MASK_BATCH=<mask选定值> RES_BATCH=<res选定值> \
 bash 'sae+momask/run.sh' all
```

## 本地检查

```bash
python -m unittest discover -s 'sae+momask/tests' -v
```

测试使用小型真实RVQ与原生Transformer，并用测试用CLIP替身避免下载大模型。
CPU测试不代替服务器CUDA/BF16、真实数据与真实评估器的短步验证。
