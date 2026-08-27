# AirNav learnable-memory online GRPO

This extension starts from `model_weight/AirNavSFT` and learns one language policy that emits a memory action and an AirNav navigation block at every segment. It never initializes from `AirVLN-R1`.

## Implemented semantics

- Four retained visual slots plus the current observation, so the model still sees at most five images.
- `APPEND_CURRENT` is the only legal warm-up action while fewer than four slots are occupied.
- At capacity, `DROP_1` through `DROP_4` replace an old slot with the current view; `DROP_CURRENT` leaves memory unchanged.
- Navigation actions are executed in NavGym before the next prompt is constructed.
- The local AirNav reward keeps format, stop, local distance progress, and yaw alignment. There is no memory-choice reward.
- Local teacher blocks are recomputed from the model's current online pose against the remaining expert trajectory. Expert segment states are never restored.
- Each episode condition is repeated `rollout.n` times. Rewards are discounted backward within `(episode, rollout)`, then normalized across live rollouts sharing `(episode, segment_index)`.
- KL loss uses the AirNavSFT reference policy.

Rollouts may terminate at different segment counts. A segment group therefore contains the rollouts still alive at that index; a singleton group receives zero advantage.

## Environment fix

The current `airnav` environment has an incompatible `tensordict==0.6.2`. Run once:

```bash
conda activate airnav
conda env config vars set PYTHONNOUSERSITE=1
python -m pip install "tensordict==0.8.3"
conda deactivate
conda activate airnav
python -m pip check
```

## Prepare episode parquet

```bash
cd /nfsdata/yangjing/AirNav
bash verl/my_script/prepare_airnav_memory_data.sh
```

## Smoke test, then train

Start with one episode and short rollouts before a full job:

```bash
cd /nfsdata/yangjing/AirNav
AIRNAV_MAX_SEGMENTS=2 TRAIN_BATCH_SIZE=1 ROLLOUT_GROUP_SIZE=5 \
bash verl/my_script/run_airnav_memory_grpo.sh vllm \
  trainer.total_training_steps=1
```

Then run the default full command:

```bash
bash verl/my_script/run_airnav_memory_grpo.sh
```

`actor_rollout_ref.rollout.agent.num_workers=1` is intentional: one worker caches the 4.2 GB orthophoto set and the GSAM map once. Increase this only after measuring host memory.

## Baseline comparison

The same evaluator supports fixed AirNav history for the downloaded SFT/R1 baselines and learned memory for the trained policy:

```bash
bash verl/my_script/eval_airnav_baselines.sh sft
bash verl/my_script/eval_airnav_baselines.sh r1

MODEL_PATH=/path/to/merged/memory-model \
bash verl/my_script/eval_airnav_baselines.sh memory
```

Results contain NE, SR, OSR and SPL under `result/`.
