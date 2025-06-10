import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from matplotlib import pyplot as plt


def time_function(func, *args, **kwargs) -> any:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start} seconds")
    return result


def time_function_out(func, *args, **kwargs) -> tuple[any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    diff = end - start
    print(f"{func.__name__} took {diff} seconds")
    return result, diff


def print_time(start, message="Elapsed time") -> None:
    end = time.perf_counter()
    print(f"{message}: {end - start} seconds")


def display_info(assignment_number: int) -> None:
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT460")
    print(f"Assignment: {assignment_number}")


def display_info_project() -> None:
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT460")
    print("Project: _")
    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("GPU Name: " + str(torch.cuda.get_device_name(0)))


def load_device() -> str:
    return (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )


def moving_average(data: list[float], window_size=10) -> np.ndarray:
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def coverage_stagnated(coverage_history: list[int], limit: int) -> bool:
    if len(coverage_history) < 100:
        return False

    return coverage_history[-1] - coverage_history[-100] < limit


def plot_statistics(episode_rewards: list[float], episode_durations: list[float], suffix: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Rewards
    ax1.plot(episode_rewards, alpha=0.4, label='Raw')
    ax1.plot(moving_average(episode_rewards), label='Moving Average')
    ax1.set_title('Episode Rewards')
    ax1.legend()

    # Durations
    ax2.plot(episode_durations, alpha=0.4, label='Raw')
    ax2.plot(moving_average(episode_durations), label='Moving Average')
    ax2.set_title('Episode Durations')
    ax2.legend()

    plt.tight_layout()

    plt.savefig(f"project/output/plots/training_progress_{suffix}.png")
    plt.close()


def save_episode_data(step_infos: list[dict[str, Any]], episode: int, run_id: str, base_path: str = "project/output/logs") -> None:
    run_folder = Path(base_path) / run_id
    run_folder.mkdir(parents=True, exist_ok=True)

    save_path = run_folder / f"episode_{episode}_data.npz"
    np.savez(save_path, data=step_infos)

    print(f"Run data saved to: {run_folder}")


def load_episode_data(episode: int, run_id: str, base_path: str = "project/output/logs") -> list[dict]:
    run_folder = Path(base_path) / run_id
    file_path = run_folder / f"episode_{episode}_data.npz"

    loaded = np.load(file_path, allow_pickle=True)
    step_infos = loaded["data"]

    return list(step_infos)


def save_metadata_json(metadata: Mapping[str, Any], run_id: str, base_path: str = "project/output/logs") -> None:
    metadata_path = Path(base_path) / run_id / "metadata.json"

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to: {metadata_path}")



