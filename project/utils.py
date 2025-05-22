import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split


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

    plt.savefig(f"project/generated/plots/training_progress_{suffix}.png")
    plt.close()

