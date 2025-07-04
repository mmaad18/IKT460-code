import json
import time
from datetime import datetime
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


def logs_path(run_id: str, base_path: str = "project/output/logs") -> Path:
    run_folder = Path(base_path) / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


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


def save_episode_data(step_infos: list[dict[str, Any]], episode: int, run_id: str) -> None:
    save_path = logs_path(run_id) / f"episode_{episode}_data.npz"
    np.savez(save_path, data=step_infos)

    print(f"Episode data saved to: {save_path}")


def load_episode_data(episode: int, run_id: str) -> list[dict]:
    file_path = logs_path(run_id) / f"episode_{episode}_data.npz"

    loaded = np.load(file_path, allow_pickle=True)
    step_infos = loaded["data"]

    return list(step_infos)


def save_metadata_json(metadata: Mapping[str, Any], run_id: str) -> None:
    metadata_path = logs_path(run_id) / "metadata.json"

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to: {metadata_path}")
    
    
def save_commentary(run_id: str) -> None:
    run_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S")
    
    comment = f"""
# Comments

### Time of run
{run_time}

### Reward function
self.reward_coefficients = np.array([
            -0.005 / self.dt,  # time
            -0.25 / self.omega_max,  # omega
            -1000.0,  # collision
            1.0 / self.v_max,  # velocity
            50.0,  # coverage
        ], dtype=np.float32)

features = np.array([
            1.0,  # time
            abs(omega),  # omega
            1.0 if _check_collision() else 0.0,  # collision
            v,  # velocity
            delta,  # coverage
        ], dtype=np.float32)
        
R = np.dot(reward_coefficients, features)
    """
    
    comment_path = logs_path(run_id) / "comment.md"
    comment_path.parent.mkdir(parents=True, exist_ok=True)

    with open(comment_path, 'w') as f:
        f.write(comment)
        
    print(f"Comment saved to: {comment_path}")



