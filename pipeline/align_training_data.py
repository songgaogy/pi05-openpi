"""
Convert the ARX A5 "put_shrimp_in_pot" dataset into the LeRobot layout used by openpi fine-tuning.

This follows the official LIBERO conversion example:
    uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/data

Example:
    uv run pipeline/align_training_data.py \
        --data_dir data/arx_a5/put_shrimp_in_pot \
        --repo_id arx_a5/put_shrimp_in_pot_openpi

The output is written to $HF_LEROBOT_HOME/<repo_id> by default, which is where openpi's training
loader will look for local LeRobot datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro


DEFAULT_REPO_ID = "arx_a5/put_shrimp_in_pot_openpi"
DEFAULT_TASK = ""

STATE_KEYS = (
    "observation.left.joint_pos",
    "observation.left.gripper_pos",
    "observation.right.joint_pos",
    "observation.right.gripper_pos",
)

ACTION_KEYS = (
    "action.left.joint_pos",
    "action.left.gripper_pos",
    "action.right.joint_pos",
    "action.right.gripper_pos",
)

CAMERA_MAP = {
    "image": "observation.images.global_front",
    "wrist_image": "observation.images.wrist_left",
    "right_wrist_image": "observation.images.wrist_right",
}


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def _load_info(data_dir: Path) -> dict:
    info_path = data_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def _load_tasks(data_dir: Path) -> dict[int, str]:
    tasks_path = data_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}

    tasks_df = pd.read_parquet(tasks_path).reset_index()
    if "task" not in tasks_df.columns or "task_index" not in tasks_df.columns:
        raise ValueError(f"{tasks_path} must contain 'task' and 'task_index' columns.")

    return {int(row["task_index"]): str(row["task"]) for _, row in tasks_df.iterrows()}


def _vector(row: pd.Series, keys: tuple[str, ...], *, dtype: np.dtype = np.float32) -> np.ndarray:
    values = []
    for key in keys:
        if key not in row:
            raise KeyError(f"Missing required column '{key}' in source parquet.")
        values.append(np.asarray(row[key], dtype=dtype).reshape(-1))
    return np.concatenate(values).astype(dtype, copy=False)


def _image_feature(source_features: dict, source_key: str) -> dict:
    if source_key not in source_features:
        raise KeyError(f"Missing camera feature '{source_key}' in meta/info.json.")

    shape = tuple(source_features[source_key]["shape"])
    if len(shape) != 3:
        raise ValueError(f"Camera feature '{source_key}' must be HWC or CHW, got {shape}.")

    return {
        "dtype": "image",
        "shape": shape,
        "names": source_features[source_key].get("names", ["height", "width", "channel"]),
    }


def _open_video(data_dir: Path, chunk_name: str, camera_key: str, episode_index: int) -> cv2.VideoCapture:
    video_path = data_dir / "videos" / chunk_name / camera_key / f"episode_{episode_index:06d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    return capture


def _read_rgb_frame(capture: cv2.VideoCapture, *, video_name: str, frame_index: int) -> np.ndarray:
    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_name}.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _make_dataset(
    *,
    repo_id: str,
    source_data_dir: Path,
    output_root: Path | None,
    fps: int,
    features: dict,
    overwrite: bool,
    image_writer_threads: int,
    image_writer_processes: int,
) -> LeRobotDataset:
    output_path = (output_root / repo_id) if output_root is not None else (HF_LEROBOT_HOME / repo_id)
    output_path = output_path.expanduser().resolve()

    if output_path == source_data_dir:
        raise ValueError(f"Refusing to overwrite the source dataset directory: {source_data_dir}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="arx_a5",
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def main(
    data_dir: Path = Path("data/arx_a5/put_shrimp_in_pot"),
    *,
    repo_id: str = DEFAULT_REPO_ID,
    output_root: Path | None = None,
    task: str = DEFAULT_TASK,
    overwrite: bool = True,
    push_to_hub: bool = False,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    info = _load_info(data_dir)
    source_features = info["features"]
    tasks = _load_tasks(data_dir)

    state_dim = sum(source_features[key]["shape"][0] for key in STATE_KEYS)
    action_dim = sum(source_features[key]["shape"][0] for key in ACTION_KEYS)

    features = {
        output_key: _image_feature(source_features, source_key)
        for output_key, source_key in CAMERA_MAP.items()
    }
    features.update(
        {
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        }
    )

    dataset = _make_dataset(
        repo_id=repo_id,
        source_data_dir=data_dir,
        output_root=output_root.expanduser().resolve() if output_root is not None else None,
        fps=int(info["fps"]),
        features=features,
        overwrite=overwrite,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    parquet_paths = sorted((data_dir / "data").glob("chunk-*/episode_*.parquet"), key=_episode_index)
    if not parquet_paths:
        raise FileNotFoundError(f"No source parquet episodes found under {data_dir / 'data'}")

    for parquet_path in tqdm(parquet_paths, desc="Converting episodes"):
        episode_index = _episode_index(parquet_path)
        chunk_name = parquet_path.parent.name
        episode_df = pd.read_parquet(parquet_path)
        task_index = int(np.asarray(episode_df["task_index"].iloc[0]).reshape(-1)[0])
        episode_task = task or tasks.get(task_index, "put_shrimp_in_pot")

        captures = {
            output_key: _open_video(data_dir, chunk_name, source_key, episode_index)
            for output_key, source_key in CAMERA_MAP.items()
        }

        try:
            for frame_index, row in episode_df.iterrows():
                frame = {
                    output_key: _read_rgb_frame(
                        capture,
                        video_name=CAMERA_MAP[output_key],
                        frame_index=int(frame_index),
                    )
                    for output_key, capture in captures.items()
                }
                frame["state"] = _vector(row, STATE_KEYS)
                frame["actions"] = _vector(row, ACTION_KEYS)
                frame["task"] = episode_task
                dataset.add_frame(frame)
        finally:
            for capture in captures.values():
                capture.release()

        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["arx_a5", "openpi", "pi05"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Saved LeRobot dataset to: {dataset.root}")
    print(f"Use repo_id in your openpi config: {repo_id}")
    print(f"state_dim={state_dim}, action_dim={action_dim}, fps={info['fps']}")


if __name__ == "__main__":
    tyro.cli(main)
