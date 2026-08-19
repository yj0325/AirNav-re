"""Evaluate AirNav policies with the official NE/SR/OSR/SPL protocol.

The learned-memory policy uses exactly the joint-action prompt and parser used
during online GRPO training.  Multiple OpenAI-compatible vLLM endpoints can be
provided to data-parallelize episode evaluation.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from navgym.models.AirNavData import AirNavData
from navgym.models.NavGym import NavGym
from navgym.tools.EvalTools import eval_planning_metrics
from train_to_sharegpt import SYSTEM_PROMPT as AIRNAV_SFT_SYSTEM_PROMPT
from train_to_sharegpt import USER_TEXT_TEMPLATE as AIRNAV_SFT_USER_TEXT_TEMPLATE
from verl.airnav_memory.actions import parse_joint_action
from verl.airnav_memory.agent_loop import build_segment_messages
from verl.airnav_memory.memory import MemoryWindow


NAV_TO_ENV = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_RIGHT": 2, "TURN_LEFT": 3}
TEST_SPLITS = ("test_easy", "test_medium", "test_hard")
_THREAD_LOCAL = threading.local()


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_from_nav(nav_gym: NavGym) -> Image.Image:
    image = np.asarray(nav_gym.rgb_crop)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return Image.fromarray(image, mode="RGB")


def learned_memory_messages(instruction, pose, actions, memory, current):
    """Materialize the training prompt's image placeholders for OpenAI API."""
    messages = build_segment_messages(
        instruction=instruction,
        pose=pose,
        history_actions=actions,
        memory_size=len(memory),
    )
    images = iter([*memory, current])
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        materialized = []
        for part in message["content"]:
            if part.get("type") == "image":
                materialized.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(next(images))}"},
                    }
                )
            else:
                materialized.append(part)
        message["content"] = materialized
    return messages


def fixed_memory_messages(instruction, pose, actions, memory, current):
    """Original AirNavSFT prompt, materialized for the OpenAI vision API."""
    content = []
    for index, frame in enumerate(memory):
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"},
                },
                {"type": "text", "text": f"(Above is historical view {index + 1})"},
            ]
        )
    content.extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(current)}"},
            },
            {"type": "text", "text": "(Above is the current view)"},
            {
                "type": "text",
                "text": AIRNAV_SFT_USER_TEXT_TEMPLATE.format(
                    instruction=instruction,
                    cur_pose=str(pose),
                    history_actions=json.dumps(actions) if actions else "None",
                ),
            },
        ]
    )
    return [
        {"role": "system", "content": AIRNAV_SFT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_fixed_action(text: str) -> tuple[str, ...]:
    response = text.replace("'", '"')
    match = re.search(r"\[(.*?)\]", response, flags=re.S)
    if match is None:
        raise ValueError("missing JSON action list")
    actions = json.loads(match.group(0))
    if not isinstance(actions, list) or not actions:
        raise ValueError("navigation action list must be non-empty")
    if not all(isinstance(action, str) and action in NAV_TO_ENV for action in actions):
        raise ValueError("invalid navigation action")
    return tuple(actions)


def get_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    cache = getattr(_THREAD_LOCAL, "clients", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.clients = cache
    key = (base_url, api_key, timeout)
    if key not in cache:
        cache[key] = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
    return cache[key]


def serialize_pose(pose) -> list[float]:
    if hasattr(pose, "tolist"):
        return pose.tolist()
    try:
        return [float(value) for value in pose]
    except TypeError:
        return [float(pose.x), float(pose.y), float(pose.z), float(pose.yaw)]


def evaluate_episode(args, dataset, episode_index, key, item, endpoint):
    nav_gym = NavGym(
        dataset.get_item(episode_index[item["episode_id"]], copy_arrays=False),
        write_images=False,
        track_visualization=False,
    )
    client = get_client(endpoint, args.api_key, args.request_timeout)
    all_views: list[Image.Image] = []
    memory = MemoryWindow(capacity=4)
    history_actions: list[str] = []
    trajectory = [nav_gym.cur_pose]
    invalid_outputs = 0
    api_errors = 0
    termination_reason = "max_actions"

    for segment_index in range(args.max_segments):
        current = image_from_nav(nav_gym)
        if args.policy == "learned":
            messages = learned_memory_messages(
                item["instruction"], nav_gym.cur_position, history_actions, memory.snapshot(), current
            )
        else:
            prompt_memory = [all_views[i] for i in (-7, -4, -2, -1) if len(all_views) >= abs(i)]
            messages = fixed_memory_messages(
                item["instruction"], nav_gym.cur_position, history_actions, prompt_memory, current
            )

        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_response_tokens,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as error:
            api_errors += 1
            response_text = ""
            if args.fail_on_api_error:
                raise RuntimeError(f"API request failed for {key} segment {segment_index}: {error}") from error

        try:
            if args.policy == "learned":
                joint_action = parse_joint_action(response_text, memory_size=len(memory.frames), capacity=4)
                memory.update(joint_action.memory_action, current)
                nav_actions = joint_action.navigation_actions
            else:
                nav_actions = parse_fixed_action(response_text)
        except Exception:
            invalid_outputs += 1
            nav_actions = ("STOP",)

        all_views.append(current)
        requested_stop = False
        for action in nav_actions:
            history_actions.append(action)
            if action == "STOP":
                requested_stop = True
                termination_reason = "stop"
                break
            nav_gym.step(NAV_TO_ENV[action], savefig=False, saveviewfig=False)
            trajectory.append(nav_gym.cur_pose)

        if requested_stop:
            break
        if len(history_actions) >= args.max_actions:
            termination_reason = "max_actions"
            break
    else:
        termination_reason = "max_segments"

    episode = dataset.episodes[episode_index[item["episode_id"]]]
    final_distance = nav_gym.cur_pose.xy.dist_to(episode.target_position.xy)
    oracle_distance = min(pose.xy.dist_to(episode.target_position.xy) for pose in trajectory)
    record = {
        "episode_id": item["episode_id"],
        "instruction": item["instruction"],
        "actions": history_actions,
        "segments": segment_index + 1,
        "end_pose": serialize_pose(nav_gym.cur_pose),
        "final_distance": float(final_distance),
        "oracle_distance": float(oracle_distance),
        "success": bool(final_distance <= 20.0),
        "oracle_success": bool(oracle_distance <= 20.0),
        "invalid_outputs": invalid_outputs,
        "api_errors": api_errors,
        "termination_reason": termination_reason,
        "endpoint": endpoint,
    }
    return key, episode, trajectory, record


def metric_dict(episodes, trajectories) -> dict[str, float]:
    metrics = eval_planning_metrics(episodes, trajectories, use_teacher_dst=True)
    return {
        "NE": float(metrics.mean_final_pos_to_goal_dist),
        "SR": float(metrics.success_rate_final_pos_to_goal),
        "OSR": float(metrics.success_rate_oracle_pos_to_goal),
        "SPL": float(metrics.success_rate_weighted_by_path_length),
    }


def eval_split(args, airnav_path, info_path, split_name):
    dataset = AirNavData(airnav_path)
    episode_index = {episode.id[-1]: index for index, episode in enumerate(dataset.episodes)}
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)
    source_items = list(info.items())
    if args.one_instruction_per_episode:
        seen_episode_ids = set()
        items = []
        for key, item in source_items:
            episode_id = item["episode_id"]
            if episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            items.append((key, item))
    else:
        items = source_items
    items = items[: args.max_episodes or None]

    print(
        json.dumps(
            {
                "event": "split_start",
                "split": split_name,
                "source_instruction_conditions": len(source_items),
                "selected_conditions": len(items),
                "one_instruction_per_episode": args.one_instruction_per_episode,
                "workers": args.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # GSamMap lazily expands a multi-GB NPZ cache on the first NavGym state.
    # Its class-level initialization is not synchronized, so letting every
    # evaluation thread hit it at once causes the same archive to be expanded
    # many times.  Warm one state on the main thread before enabling parallelism.
    if items:
        warm_episode_id = items[0][1]["episode_id"]
        warm_nav_gym = NavGym(
            dataset.get_item(episode_index[warm_episode_id], copy_arrays=False),
            write_images=False,
            track_visualization=False,
        )
        del warm_nav_gym

    episodes, trajectories, records = {}, {}, {}
    start_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_episode,
                args,
                dataset,
                episode_index,
                key,
                item,
                args.base_url[index % len(args.base_url)],
            ): key
            for index, (key, item) in enumerate(items)
        }
        for completed, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc=split_name), start=1
        ):
            key, episode, trajectory, record = future.result()
            episodes[key] = episode
            trajectories[key] = trajectory
            records[key] = record
            if completed % args.progress_log_interval == 0 or completed == len(futures):
                elapsed_seconds = time.monotonic() - start_time
                conditions_per_second = completed / max(elapsed_seconds, 1e-9)
                remaining = len(futures) - completed
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "split": split_name,
                            "completed": completed,
                            "total": len(futures),
                            "percent": round(100.0 * completed / max(len(futures), 1), 2),
                            "conditions_per_second": round(conditions_per_second, 3),
                            "elapsed_seconds": round(elapsed_seconds, 1),
                            "eta_seconds": round(remaining / max(conditions_per_second, 1e-9), 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    metrics = metric_dict(episodes, trajectories)
    result = {
        "split": split_name,
        "model": args.model_name,
        "policy": args.policy,
        "temperature": args.temperature,
        "one_instruction_per_episode": args.one_instruction_per_episode,
        "source_instruction_conditions": len(source_items),
        "num_conditions": len(records),
        "metrics": metrics,
        "diagnostics": {
            "format_valid_rate": float(
                np.mean([record["invalid_outputs"] == 0 for record in records.values()])
            ),
            "api_error_count": int(sum(record["api_errors"] for record in records.values())),
            "mean_segments": float(np.mean([record["segments"] for record in records.values()])),
        },
        "episodes": records,
    }
    return result, episodes, trajectories


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="airnav")
    parser.add_argument(
        "--base-url",
        action="append",
        help="OpenAI-compatible endpoint; repeat to data-parallelize across vLLM servers",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--policy", choices=["fixed", "learned"], default="learned")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-actions", type=int, default=160)
    parser.add_argument("--max-segments", type=int, default=160)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument(
        "--one-instruction-per-episode",
        action="store_true",
        help="Deterministically evaluate only the first instruction for each physical episode.",
    )
    parser.add_argument("--progress-log-interval", type=int, default=50)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--fail-on-api-error", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=["val_seen", "val_unseen", "test_easy", "test_medium", "test_hard"],
    )
    args = parser.parse_args()
    if args.progress_log_interval <= 0:
        parser.error("--progress-log-interval must be positive")
    if not args.base_url:
        args.base_url = ["http://localhost:8000/v1"]

    root = Path(__file__).resolve().parent
    split_paths = {
        "val_seen": ("val/airnav_val_seen.json", "val/info_val_seen.json"),
        "val_unseen": ("val/airnav_val_unseen.json", "val/info_val_unseen.json"),
        "test_easy": ("test/airnav_test_easy.json", "test/info_test_easy.json"),
        "test_medium": ("test/airnav_test_medium.json", "test/info_test_medium.json"),
        "test_hard": ("test/airnav_test_hard.json", "test/info_test_hard.json"),
    }
    selected = args.split or list(split_paths)
    results = []
    test_episodes, test_trajectories = {}, {}
    output = Path(args.output)

    for split in selected:
        airnav_rel, info_rel = split_paths[split]
        result, episodes, trajectories = eval_split(
            args,
            root / "data/AirNav" / airnav_rel,
            root / "data/AirNav" / info_rel,
            split,
        )
        results.append(result)
        print(json.dumps({"split": split, **result["metrics"], **result["diagnostics"]}, indent=2))
        if split in TEST_SPLITS:
            test_episodes.update({f"{split}:{key}": value for key, value in episodes.items()})
            test_trajectories.update({f"{split}:{key}": value for key, value in trajectories.items()})
        write_results(output, results)

    if all(split in selected for split in TEST_SPLITS):
        aggregate = {
            "split": "test_unseen",
            "model": args.model_name,
            "policy": args.policy,
            "temperature": args.temperature,
            "num_conditions": len(test_episodes),
            "metrics": metric_dict(test_episodes, test_trajectories),
        }
        results.append(aggregate)
        print(json.dumps({"split": "test_unseen", **aggregate["metrics"]}, indent=2))
        write_results(output, results)


if __name__ == "__main__":
    main()
