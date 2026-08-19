#!/usr/bin/env python3
"""Build the missing AirNav GRPO parquet data from the released JSON files.

The upstream repository generated every persona independently and repeatedly
rewrote one very large JSON file.  This implementation keeps the released
prompt/reward schema, but generates each physical episode once, reuses its
views for all persona instructions, and writes resumable parquet shards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

matplotlib.use("Agg")


SYSTEM_PROMPT = """
## Role
You are an expert navigation assistant for a UAV (Unmanned Aerial Vehicle) flight simulator.

## Task Objective
The UAV operates in a 3D urban environment with visible roads, buildings, and landmarks.
Your task is to predict the next sequence of UAV actions based on:
1. A given natural language navigation instruction,
2. The current state of the UAV, including its position and heading angle,
3. The current top-down UAV view image,
4. Up to four historical top-down view images from previous time steps (if available),
5. The previously executed UAV actions (if available).
"""

USER_PROMPT = """
## Text Input
- **Navigation instruction**: {instruction}  
- **Current state of the UAV**: {cur_pose} (x, y, z in meters; heading in degrees)
- **Previously executed actions**: {history_actions}  
  (A list of past actions the UAV has taken, in chronological order.)

## Image Input
**UAV (Unmanned Aerial Vehicle) Top-Down View Sequence**  
- Historical top-down views (from oldest to newest) show the UAV’s past observations. 
- The last image is the current top-down view of the UAV.  
- In all images, the **top of the image corresponds to the UAV’s forward direction** (its heading).

Based on the navigation instruction, the UAV’s current state, the previously executed actions (which can help infer the UAV’s current orientation and progress), and the provided images, predict how the UAV should move **step by step** to follow the instruction accurately.

## Prediction Rules
1. Predict no more than **8 future actions** for the UAV to execute.
2. If the target location is reachable in fewer than 8 actions, output less than 8 actions sequence and end with **"STOP"**. Otherwise, it clearly requires more than 8 actions to approach the target, output exactly 8 future actions.
3. You **must** output **"STOP"** if the UAV has already reached the described target.
4. Output a **JSON list** of actions, in the **exact order** they should be executed.
5. Do **not** include any explanations, reasoning, or additional text — only output the JSON list.

## Discrete Action Space
- `MOVE_FORWARD`: move straight 5 meters in the current heading
- `TURN_LEFT`: rotate left 30°
- `TURN_RIGHT`: rotate right 30°
- `STOP`: stop the flight

## Output Format Examples
["TURN_RIGHT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"]  
or  
["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]  
or  
["STOP"]
"""

HISTORY_INDICES = (-7, -4, -2, -1)
ACTION_CODE = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_RIGHT": 2, "TURN_LEFT": 3}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_prompt(instruction: str, cur_pose: list[float], history_actions: list[str], history_count: int) -> list[dict]:
    image_prefix = "".join(
        f"<image>\n(Above is historical view {index + 1})\n" for index in range(history_count)
    )
    image_prefix += "<image>\n(Above is the current view)\n"
    user = image_prefix + USER_PROMPT.format(
        instruction=instruction,
        cur_pose=cur_pose,
        history_actions=json.dumps(history_actions) if history_actions else "None",
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def group_instructions(info_path: Path) -> tuple[list[str], dict[str, list[tuple[str, dict]]]]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    order: list[str] = []
    for key, value in info.items():
        episode_id = value["episode_id"]
        if episode_id not in grouped:
            order.append(episode_id)
        grouped[episode_id].append((key, value))
    return order, grouped


def load_episode_index(airnav_path: Path) -> dict[str, dict]:
    records = json.loads(airnav_path.read_text(encoding="utf-8"))
    result = {}
    for record in records:
        episode_id = f"{record['area']}_block_{record['block']}_{record['object_ids'][0]}_{record['ann_ids'][0]}"
        result[episode_id] = record
    return result


def build_map_cache(root: Path, records: dict[str, dict]) -> dict[str, tuple[rasterio.DatasetReader, object]]:
    map_names = sorted({f"{record['area']}_block_{record['block']}" for record in records.values()})
    cache = {}
    for map_name in map_names:
        raster = rasterio.open(root / "data/rgbd-new" / f"{map_name}.tif")
        image = cv2.cvtColor(cv2.imread(str(root / "data/rgbd-new" / f"{map_name}.png")), cv2.COLOR_BGR2RGB)
        cache[map_name] = (raster, image)
    return cache


def sample_history(paths: list[str]) -> list[str]:
    return [paths[index] for index in HISTORY_INDICES if len(paths) >= abs(index)]


def make_row(case: dict, instruction_key: str, instruction: dict, split: str, row_index: int) -> dict:
    history = sample_history(case["history_views"])
    images = history + [case["cur_view"]]
    prompt = build_prompt(instruction["instruction"], case["cur_pose"], case["history_actions"], len(history))
    return {
        "data_source": "AirNav_rl",
        "prompt": prompt,
        "images": [{"image": image_path} for image_path in images],
        "ability": "navigation",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(case["future_actions"])},
        "extra_info": {
            "split": split,
            "index": row_index,
            "instruction_key": instruction_key,
            "episode_id": instruction["episode_id"],
            "target_px": case["target_px"],
            "map_name": case["map_name"],
            "cur_px": case["cur_px"],
            "cur_pose": case["cur_pose"],
        },
    }


def render_episode(
    record: dict,
    episode_id: str,
    actions: list[str],
    image_root: Path,
    map_cache: dict[str, tuple[rasterio.DatasetReader, object]],
) -> list[dict]:
    from gsamllavanav.mapdata import GROUND_LEVEL
    from gsamllavanav.space import Pose4D
    from navgym.tools.ImgTools import crop_rpg

    map_name = f"{record['area']}_block_{record['block']}"
    raster, image = map_cache[map_name]
    x, y, _, yaw, *_ = record["trajectory"][0]
    pose = Pose4D(x, y, GROUND_LEVEL[map_name] + 50, yaw)
    view_pixels = int((pose.z - GROUND_LEVEL[map_name]) / abs(raster.transform.a))
    view_shape = (view_pixels, view_pixels)
    view_real_size = view_pixels * abs(raster.transform.a)
    target_x, target_y, _ = record["target_positions"][-1]
    target_row, target_col = raster.index(target_x, target_y)
    target_px = [int(target_col), int(target_row)]
    episode_dir = image_root / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    history_views: list[str] = []
    history_actions: list[str] = []
    cases: list[dict] = []
    for case_index, offset in enumerate(range(0, len(actions), 8)):
        view_path = (episode_dir / f"case{case_index}.jpg").resolve()
        if not view_path.exists():
            view, _ = crop_rpg(
                image=image,
                pose=pose,
                shape=view_shape,
                raster=raster,
                map_name=map_name,
                shape_real_size=view_real_size,
            )
            cv2.imwrite(str(view_path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        row, col = raster.index(pose.x, pose.y)
        future = actions[offset : offset + 8]
        cases.append(
            {
                "history_views": history_views.copy(),
                "cur_view": str(view_path),
                "history_actions": history_actions.copy(),
                "future_actions": future,
                "cur_pose": [pose.x, pose.y, pose.z, math.degrees(pose.yaw)],
                "cur_px": [int(col), int(row)],
                "target_px": target_px,
                "map_name": map_name,
            }
        )
        history_views.append(str(view_path))
        history_actions.extend(future)
        for action in future:
            code = ACTION_CODE[action]
            if code == 1:
                pose = Pose4D(pose.x + 5 * math.cos(pose.yaw), pose.y + 5 * math.sin(pose.yaw), pose.z, pose.yaw)
            elif code == 3:
                pose = Pose4D(pose.x, pose.y, pose.z, (pose.yaw + math.pi / 6 + math.pi) % (2 * math.pi) - math.pi)
            elif code == 2:
                pose = Pose4D(pose.x, pose.y, pose.z, (pose.yaw - math.pi / 6 + math.pi) % (2 * math.pi) - math.pi)
    return cases


def prepare_split(
    root: Path,
    split: str,
    info_path: Path,
    airnav_path: Path,
    output_dir: Path,
    image_root: Path,
    episodes_per_shard: int,
    limit_episodes: int,
    clean: bool,
) -> None:
    split_dir = output_dir / split
    progress_path = split_dir / "progress.json"
    if clean and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    episode_order, grouped = group_instructions(info_path)
    records = load_episode_index(airnav_path)
    if limit_episodes:
        episode_order = episode_order[:limit_episodes]
        records = {episode_id: records[episode_id] for episode_id in episode_order}
    map_cache = build_map_cache(root, records)

    complete_shards: list[int] = []
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        complete_shards = state.get("complete_shards", [])

    total_shards = math.ceil(len(episode_order) / episodes_per_shard)
    row_index = 0
    try:
        for shard_index in range(total_shards):
            start = shard_index * episodes_per_shard
            end = min(start + episodes_per_shard, len(episode_order))
            output_path = split_dir / f"part-{shard_index:05d}.parquet"
            expected_done = shard_index in complete_shards and output_path.exists()
            if expected_done:
                row_index += pq.read_metadata(output_path).num_rows
                print(f"[{split}] shard {shard_index + 1}/{total_shards} already complete", flush=True)
                continue

            rows = []
            for local_index, episode_id in enumerate(episode_order[start:end], 1):
                if episode_id not in records:
                    raise KeyError(f"Episode {episode_id} is missing from {airnav_path}")
                candidates = grouped[episode_id]
                action_variants = {json.dumps(item[1]["total_actions"]) for item in candidates}
                if len(action_variants) != 1:
                    raise ValueError(f"Persona action mismatch for {episode_id}")
                cases = render_episode(
                    records[episode_id], episode_id, candidates[0][1]["total_actions"], image_root, map_cache
                )
                for instruction_key, instruction in candidates:
                    for case in cases:
                        rows.append(make_row(case, instruction_key, instruction, split, row_index + len(rows)))
                if local_index % 25 == 0 or local_index == end - start:
                    print(
                        f"[{split}] shard {shard_index + 1}/{total_shards}: "
                        f"episodes {local_index}/{end - start}, rows {len(rows)}",
                        flush=True,
                    )

            temporary = output_path.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
            os.replace(temporary, output_path)
            complete_shards.append(shard_index)
            row_index += len(rows)
            atomic_json(
                progress_path,
                {
                    "split": split,
                    "episodes": len(episode_order),
                    "total_shards": total_shards,
                    "complete_shards": sorted(set(complete_shards)),
                    "rows_written": row_index,
                },
            )
            print(f"[{split}] wrote {len(rows)} rows to {output_path}", flush=True)
    finally:
        for raster, _ in map_cache.values():
            raster.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("train", "val_seen", "all"), default="all")
    parser.add_argument("--episodes-per-shard", type=int, default=250)
    parser.add_argument("--limit-episodes", type=int, default=0, help="Smoke-test cap; zero means all episodes")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    output_dir = root / "data/AirNav_GRPO"
    image_root = root / "TrainPhotoData"
    selections = ["train", "val_seen"] if args.split == "all" else [args.split]
    paths = {
        "train": (root / "data/AirNav/train/info_train.json", root / "data/AirNav/train/airnav_train.json"),
        "val_seen": (root / "data/AirNav/val/info_val_seen.json", root / "data/AirNav/val/airnav_val_seen.json"),
    }
    for split in selections:
        info_path, airnav_path = paths[split]
        prepare_split(
            root,
            split,
            info_path,
            airnav_path,
            output_dir,
            image_root,
            args.episodes_per_shard,
            args.limit_episodes,
            args.clean,
        )


if __name__ == "__main__":
    main()
