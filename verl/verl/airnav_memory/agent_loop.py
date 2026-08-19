"""Complete online AirNav episode rollout with a learnable four-frame memory."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from gsamllavanav.space import Pose4D
from navgym.models.AirNavData import AirNavData
from navgym.models.NavGym import NavGym
from verl.airnav_memory.actions import parse_joint_action
from verl.airnav_memory.memory import MemoryWindow
from verl.airnav_memory.reward import (
    ACTION_TO_ENV,
    RewardBreakdown,
    compute_segment_reward,
    compute_terminal_reward,
    dynamic_teacher_actions,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.utils.dataset.vision_utils import process_image


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """## Role
You are an expert navigation and memory-control assistant for a UAV flight simulator.

At every decision segment you receive the complete navigation instruction, the current UAV pose,
up to four retained historical first-person top-down views, the current view, and past navigation actions.
Jointly decide (1) which observation remains in the fixed four-slot visual memory and (2) the next
AirNav navigation action block. Memory has no externally supplied correct answer; choose it only to
improve future navigation.
"""


def build_segment_messages(
    instruction: str,
    pose: list[float],
    history_actions: list[str],
    memory_size: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    for slot in range(memory_size):
        content.extend(
            [
                {"type": "image"},
                {"type": "text", "text": f"The image above is retained memory slot {slot + 1}.\n"},
            ]
        )
    content.extend(
        [
            {"type": "image"},
            {"type": "text", "text": "The image above is the current UAV view.\n"},
        ]
    )

    if memory_size < 4:
        memory_rule = (
            f"The memory currently contains {memory_size}/4 frames. You must use APPEND_CURRENT; "
            "the current frame will be appended automatically."
        )
    else:
        memory_rule = (
            "The memory is full. Choose DROP_1, DROP_2, DROP_3, or DROP_4 to delete that old slot "
            "and append the current frame; choose DROP_CURRENT to keep all four old slots."
        )

    content.append(
        {
            "type": "text",
            "text": f"""## Text Input
- Navigation instruction: {instruction}
- Current UAV pose: {pose} (x, y, z in meters; heading in degrees)
- Previously executed navigation actions: {json.dumps(history_actions) if history_actions else "None"}

## Memory Rule
{memory_rule}

## Navigation Rules
1. Generate 1 to 8 actions from MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.
2. STOP may only be the last action. Stop if the target has been reached.
3. If more than 8 actions are needed, output exactly 8 actions for this segment.

Return only this exact JSON object, without markdown or explanation:
{{"memory_action":"APPEND_CURRENT or DROP_1/DROP_2/DROP_3/DROP_4/DROP_CURRENT","navigation_actions":["MOVE_FORWARD", "..."]}}
""",
        }
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


class AirNavMemoryAgentLoop(AgentLoopBase):
    """One loop instance rolls a model-controlled NavGym episode to termination."""

    _datasets: dict[str, AirNavData] = {}
    _episode_indices: dict[str, dict[str, int]] = {}
    _dataset_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def __init__(
        self,
        *args,
        max_episode_actions: int = 160,
        max_segments: int = 24,
        memory_capacity: int = 4,
        success_distance: float = 20.0,
        success_reward: float = 5.0,
        failure_reward: float = -2.0,
        segment_penalty: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # OmegaConf's ``oc.env`` resolver returns strings, even when the YAML
        # fallback is numeric.  Normalize these knobs once at construction so
        # both the default configuration and environment overrides behave the
        # same way inside Ray workers.
        self.max_episode_actions = int(max_episode_actions)
        self.max_segments = int(max_segments)
        self.memory_capacity = int(memory_capacity)
        self.success_distance = float(success_distance)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.segment_penalty = float(segment_penalty)
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def _get_dataset(self, path: str) -> tuple[AirNavData, dict[str, int]]:
        normalized = str(Path(path).resolve())
        if normalized not in self._datasets:
            async with self._dataset_locks[normalized]:
                if normalized not in self._datasets:
                    image_dir = str((Path(normalized).parents[2] / "rgbd-new").resolve())
                    dataset = await self.loop.run_in_executor(
                        None,
                        lambda: AirNavData(
                            normalized,
                            image_dir=image_dir,
                            lazy_maps=True,
                            map_cache_size=16,
                            lazy_images=True,
                            image_cache_size=4,
                        ),
                    )
                    self._datasets[normalized] = dataset
                    self._episode_indices[normalized] = {
                        episode.id[-1]: index for index, episode in enumerate(dataset.episodes)
                    }
        return self._datasets[normalized], self._episode_indices[normalized]

    async def _encode_prompt(self, messages: list[dict], images: list[Image.Image]) -> list[int]:
        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        model_inputs = await self.loop.run_in_executor(
            None,
            lambda: self.processor(text=[raw_prompt], images=images, return_tensors="pt"),
        )
        prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        prompt_limit = self.config.actor_rollout_ref.rollout.prompt_length
        if len(prompt_ids) > prompt_limit:
            raise RuntimeError(f"AirNav segment prompt has {len(prompt_ids)} tokens, limit is {prompt_limit}")
        return prompt_ids

    @staticmethod
    def _current_image(nav_gym: NavGym) -> Image.Image:
        image = np.asarray(nav_gym.rgb_crop)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return process_image({"image": Image.fromarray(image, mode="RGB")})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        extra_info = dict(kwargs["extra_info"])
        trajectory_info = kwargs.get("_trajectory", {})
        rollout_id = int(trajectory_info.get("rollout_n", 0))
        episode_uid = str(kwargs.get("uid", extra_info["episode_key"]))
        dataset, episode_index = await self._get_dataset(extra_info["airnav_json_path"])
        if extra_info["episode_id"] not in episode_index:
            raise KeyError(f"episode {extra_info['episode_id']} is missing from {extra_info['airnav_json_path']}")

        single_data = dataset.get_item(episode_index[extra_info["episode_id"]], copy_arrays=False)
        nav_gym = NavGym(single_data, data_dir=os.devnull, write_images=False, track_visualization=False)
        memory = MemoryWindow(capacity=self.memory_capacity)
        history_actions: list[str] = []
        segment_outputs: list[AgentLoopOutput] = []
        request_prefix = uuid4().hex
        total_actions = 0

        episode_metrics = AgentLoopMetrics()
        for segment_index in range(self.max_segments):
            current_image = self._current_image(nav_gym)
            images = memory.snapshot() + [current_image]
            messages = build_segment_messages(
                instruction=extra_info["instruction"],
                pose=nav_gym.cur_position,
                history_actions=history_actions,
                memory_size=len(memory.frames),
            )
            prompt_ids = await self._encode_prompt(messages, images)
            started = time.perf_counter()
            generated = await self.server_manager.generate(
                request_id=f"{request_prefix}-{segment_index}",
                prompt_ids=prompt_ids,
                sampling_params=dict(sampling_params),
                image_data=images,
            )
            elapsed = time.perf_counter() - started
            episode_metrics.generate_sequences += elapsed
            response_ids = generated.token_ids[: self.response_length]
            response_logprobs = (
                generated.log_probs[: self.response_length] if generated.log_probs is not None else None
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Keep the original AirNav shaping definition anchored at the
            # segment start, but only score the segment after its actions have
            # actually been executed in NavGym.
            segment_start_pose = Pose4D(*nav_gym.cur_pose)
            teacher_actions = dynamic_teacher_actions(segment_start_pose, nav_gym.episode.teacher_trajectory)
            format_valid = True
            parse_error = ""
            try:
                joint_action = parse_joint_action(
                    response_text,
                    memory_size=len(memory.frames),
                    capacity=self.memory_capacity,
                )
                predicted_actions = list(joint_action.navigation_actions)
            except (ValueError, json.JSONDecodeError, TypeError) as error:
                format_valid = False
                parse_error = str(error)
                joint_action = None
                predicted_actions = []

            terminal = False
            success = False
            termination_reason = "running"
            requested_stop = False
            executed_actions: list[str] = []

            if not format_valid or not response_ids:
                terminal = True
                termination_reason = "invalid_format" if not format_valid else "empty_response"
            else:
                memory.update(joint_action.memory_action, current_image)
                for action in predicted_actions:
                    history_actions.append(action)
                    total_actions += 1
                    if action == "STOP":
                        requested_stop = True
                        break
                    nav_gym.step(ACTION_TO_ENV[action], savefig=False, saveviewfig=False)
                    executed_actions.append(action)
                    if total_actions >= self.max_episode_actions:
                        terminal = True
                        termination_reason = "max_actions"
                        break

                final_distance = nav_gym.cur_pose.xy.dist_to(nav_gym.episode.target_position.xy)
                if requested_stop:
                    terminal = True
                    success = final_distance <= self.success_distance
                    termination_reason = "success" if success else "wrong_stop"
                elif not terminal and segment_index + 1 >= self.max_segments:
                    terminal = True
                    termination_reason = "max_segments"

            final_distance = nav_gym.cur_pose.xy.dist_to(nav_gym.episode.target_position.xy)
            shaping_reward = compute_segment_reward(
                current_pose=segment_start_pose,
                predicted_actions=predicted_actions,
                teacher_actions=teacher_actions,
                raster=nav_gym.raster,
                format_valid=format_valid,
                segment_penalty=self.segment_penalty,
            )
            terminal_value = compute_terminal_reward(
                terminal=terminal,
                success=success,
                success_reward=self.success_reward,
                failure_reward=self.failure_reward,
            )
            reward = RewardBreakdown(
                distance=shaping_reward.distance,
                yaw=shaping_reward.yaw,
                format=shaping_reward.format,
                segment=shaping_reward.segment,
                terminal=terminal_value,
            )
            segment_extra_info = {
                **extra_info,
                "segment_index": segment_index,
                "rollout_id": rollout_id,
                "teacher_actions": teacher_actions,
                "format_valid": format_valid,
                "parse_error": parse_error,
                "reward_distance": reward.distance,
                "reward_yaw": reward.yaw,
                "reward_format": reward.format,
                "reward_segment": reward.segment,
                "reward_terminal": reward.terminal,
                "episode_terminal": terminal,
                "episode_success": success,
                "termination_reason": termination_reason,
                "final_distance_to_goal": final_distance,
                "requested_stop": requested_stop,
                "predicted_action_count": len(predicted_actions),
                "executed_action_count": len(executed_actions),
                "episode_action_count": total_actions,
            }
            segment_outputs.append(
                AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=[1] * len(response_ids),
                    response_logprobs=response_logprobs,
                    multi_modal_data={"image": images},
                    reward_score=reward.total,
                    num_turns=2,
                    metrics=episode_metrics,
                    extra_fields={
                        "episode_expanded": True,
                        "episode_uid": episode_uid,
                        "rollout_id": rollout_id,
                        "segment_index": segment_index,
                        "uid": f"{episode_uid}:segment:{segment_index}",
                        "data_source": "airnav_memory",
                        "episode_terminal": terminal,
                        "episode_success": success,
                        "termination_reason": termination_reason,
                        "predicted_action_count": len(predicted_actions),
                        "executed_action_count": len(executed_actions),
                        "episode_action_count": total_actions,
                        "reward_model": {"style": "rule", "ground_truth": json.dumps(teacher_actions)},
                        "extra_info": segment_extra_info,
                    },
                )
            )

            if terminal:
                break

        return segment_outputs
