# AirNav 私有迁移包

迁移使用三个私有仓库：

- 源码：`https://github.com/yj0325/AirNav-re`
- 模型：ModelScope 模型仓库 `ddbcdd/airnav`
- 数据、完整训练 checkpoint 和环境：ModelScope 数据集仓库 `ddbcdd/airnav`

GitHub 只保存源码、实验配置、环境清单和迁移脚本。大模型、训练数据、
checkpoint 和 Conda 压缩包均不进入 Git，以避免 GitHub 单文件大小限制。

ModelScope 模型仓库保存当前冷启动模型 `AirNavMemoryColdStart/`。数据集仓库按项目根目录布局保存：

- `data/`
- `archives/TrainPhotoData/`（约4GB一卷的无损 tar 分卷）
- `archives/LargeData/`（3.21GB gsam 文件的1GB无损分卷）
- `data/airnav_memory/coldstart_10k/`（10k 冷启动 JSON 与图像）
- `checkpoints/airnav_memory_grpo/airnav_memory_grpo_coldstart_8gpu_b4_g4_u85/global_step_113/`
- `checkpoints/airnav_memory_grpo/airnav_memory_grpo_coldstart_8gpu_b4_g4_u85/latest_checkpointed_iteration.txt`
- `environment/airnav-conda-pack.tar.gz`
- `environment/SHA256SUMS`

step113 checkpoint 包含模型、优化器和 extra-state 的 8 卡 FSDP 分片，约 62GB；
它用于无损续训，因此放在数据集仓库，而不是只保存可推理权重的模型仓库。

## 源服务器

在当前服务器上确认 `modelscope whoami` 显示 `ddbcdd`，然后执行：

```bash
bash migration/package_environment.sh
bash migration/package_data.sh
bash migration/upload_modelscope.sh
```

脚本默认上传当前冷启动模型、`data/`（不含未分片的 3.2GB gsam 文件）、图片/大文件分片、
step113 checkpoint 和环境包。失败后可单独重试，例如：

```bash
UPLOAD_BASE_MODEL=0 UPLOAD_DATA=0 UPLOAD_ENV=0 bash migration/upload_modelscope.sh
```

各仓库的写入均支持 ModelScope 的断点缓存；上传完成后用 `modelscope info ddbcdd/airnav --repo-type dataset` 检查文件。

## 新服务器

```bash
python3 -m venv /tmp/airnav-transfer
/tmp/airnav-transfer/bin/pip install modelscope
/tmp/airnav-transfer/bin/modelscope login

git clone https://github.com/yj0325/AirNav-re.git AirNav
cd AirNav
PATH="/tmp/airnav-transfer/bin:$PATH" \
AIRNAV_ROOT="$PWD" \
CONDA_ROOT=/你的/miniconda3/路径 \
bash migration/download_and_restore.sh
```

恢复脚本会下载私有 ModelScope 仓库、校验并解开训练图片分卷、重建 gsam 文件、
校验环境压缩包、执行 `conda-unpack`、重写源码和 Parquet 中的旧绝对路径、重新安装本地 VERL，
并检查 CUDA 和关键包。图片分卷在成功解包后仍保留于 `archives/`，确认恢复无误后
可手动删除以释放约17GB空间。
目标机仍需安装与当前 PyTorch/CUDA兼容的 NVIDIA 驱动；打包环境不包含显卡驱动。
恢复后的训练入口直接调用打包环境的 Python，不依赖 base Conda 的安装位置。

## 恢复后启动/续训

```bash
source /你的/miniconda3/envs/airnav/bin/activate
cd AirNav
export AIRNAV_ROOT="$PWD"
export AIRNAV_MEMORY_ACTION_MASK=1
export MODEL_PATH="$PWD/model_weight/AirNavMemoryColdStart"
export CHECKPOINT_ROOT="$PWD/verl/checkpoints/airnav_memory_grpo/airnav_memory_grpo_coldstart_8gpu_b4_g4_u85"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash verl/my_script/run_airnav_memory_grpo.sh vllm \
  trainer.resume_mode=auto \
  trainer.total_training_steps=1500
```

如只做推理，直接使用 `model_weight/AirNavMemoryColdStart`；如需从 step113 继续训练，
保持 `trainer.resume_mode=auto` 和上述 checkpoint 目录即可。启动前应确认新机器有兼容的
NVIDIA 驱动、至少 8 张可用 GPU，以及足够的磁盘空间（模型约16GB，checkpoint约62GB，
环境约4.7GB，数据与图片约20GB）。
