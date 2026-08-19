# AirNav 私有迁移包

迁移使用三个私有仓库：

- 源码：`https://github.com/yj0325/AirNav`
- 模型：ModelScope 模型仓库 `ddbcdd/airnav`
- 数据、完整训练 checkpoint 和环境：ModelScope 数据集仓库 `ddbcdd/airnav`

GitHub 只保存源码、实验配置、环境清单和迁移脚本。大模型、训练数据、
checkpoint 和 Conda 压缩包均不进入 Git，以避免 GitHub 单文件大小限制。

ModelScope 模型仓库保存 `AirNavSFT/`。数据集仓库按项目根目录布局保存：

- `data/`
- `official_grpo_repro/TrainPhotoData/`
- `official_grpo_repro/checkpoints/airnav_grpo_7gpu_resume17_optimized/global_step_150/`
- `official_grpo_repro/checkpoints/airnav_grpo_7gpu_resume17_optimized/latest_checkpointed_iteration.txt`
- `environment/airnav-conda-pack.tar.gz`
- `environment/SHA256SUMS`

完整 step150 checkpoint 包含模型、优化器和 extra-state 的 7 卡 FSDP 分片，约
47GB；它用于无损续训，因此放在数据集仓库，而不是只保存可推理权重的模型仓库。

## 源服务器

先在一个小型传输环境中安装 `conda-pack` 和 `modelscope`，登录 ModelScope，
然后执行：

```bash
bash migration/package_environment.sh
bash migration/upload_modelscope.sh
```

各部分上传开关为 `UPLOAD_BASE_MODEL`、`UPLOAD_DATA`、`UPLOAD_ENV` 和
`UPLOAD_LATEST_CHECKPOINT`，默认均为 1。失败后可只重传对应部分。

## 新服务器

```bash
python3 -m venv /tmp/airnav-transfer
/tmp/airnav-transfer/bin/pip install modelscope
/tmp/airnav-transfer/bin/modelscope login

git clone https://gh-proxy.com/https://github.com/yj0325/AirNav
cd AirNav
PATH="/tmp/airnav-transfer/bin:$PATH" \
AIRNAV_ROOT="$PWD" \
CONDA_ROOT=/你的/miniconda3/路径 \
bash migration/download_and_restore.sh
```

恢复脚本会下载私有 ModelScope 仓库、校验环境压缩包、执行 `conda-unpack`、
重写源码和 Parquet 中的旧绝对路径、重新安装本地 VERL，并检查 CUDA 和关键包。
目标机仍需安装与当前 PyTorch/CUDA兼容的 NVIDIA 驱动；打包环境不包含显卡驱动。
恢复后的训练入口直接调用打包环境的 Python，不依赖 base Conda 的安装位置。
