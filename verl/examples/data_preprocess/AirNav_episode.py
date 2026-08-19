"""Create episode-level parquet files for online AirNav memory GRPO."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from datasets import Dataset


PLACEHOLDER_PROMPT = [
    {
        "role": "user",
        "content": "Initialize this AirNav episode. The online agent loop will construct each visual segment prompt.",
    }
]


def build_rows(
    info_path: Path,
    airnav_path: Path,
    split: str,
    one_instruction_per_episode: bool = False,
    seed: int = 1,
) -> list[dict]:
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)

    items = list(info.items())
    if one_instruction_per_episode:
        by_episode: dict[str, list[tuple[str, dict]]] = {}
        for episode_key, item in items:
            by_episode.setdefault(item["episode_id"], []).append((episode_key, item))

        rng = random.Random(seed)
        # Sort before sampling so selection is reproducible even if JSON key order changes.
        items = [
            rng.choice(sorted(candidates, key=lambda candidate: candidate[0]))
            for _, candidates in sorted(by_episode.items())
        ]

    rows = []
    for index, (episode_key, item) in enumerate(items):
        rows.append(
            {
                "data_source": "airnav_memory",
                "agent_name": "airnav_memory",
                "prompt": PLACEHOLDER_PROMPT,
                "ability": "navigation",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "episode_key": episode_key,
                    "episode_id": item["episode_id"],
                    "instruction": item["instruction"],
                    "airnav_json_path": str(airnav_path.resolve()),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True)
    parser.add_argument("--airnav", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--one-instruction-per-episode",
        action="store_true",
        help="Randomly retain one instruction condition for each physical episode.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        Path(args.info),
        Path(args.airnav),
        args.split,
        one_instruction_per_episode=args.one_instruction_per_episode,
        seed=args.seed,
    )
    Dataset.from_list(rows).to_parquet(str(output))
    print(f"wrote {len(rows)} episode conditions to {output}")


if __name__ == "__main__":
    main()
